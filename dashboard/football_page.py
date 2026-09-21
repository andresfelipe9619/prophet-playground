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

from dashboard import betlog_page
from dashboard.ui import HELP, chart, glossary, plain_verdict, section
from football.backtest import MODEL_NAMES
from football.calibration import (
    calibration_in_the_large,
    expected_calibration_error,
    reliability_curve,
)
from football.common import (
    DEFAULT_DATA_DIR,
    ODDS_COLUMNS,
    OUTCOMES,
    PROBABILITY_COLUMNS,
    UnknownTeamError,
)
from football.ensemble import POOLS
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
from football.value import DEFAULT_KELLY_FRACTION, DISAGREEMENT_ONLY, NO_VALUE, VALUE

# Spanish labels for the three outcomes, since OUTCOME_LABELS is English (it is
# code-facing). The order follows OUTCOMES, which is load-bearing everywhere.
OUTCOME_ES = {"H": "Local", "D": "Empate", "A": "Visitante"}

# Spanish labels for the code-facing model and verdict names.
MODEL_ES = {"dixon_coles": "Dixon-Coles", "elo": "Elo", "blend": "Mezcla con el mercado"}
VERDICT_ES = {
    VALUE: "Valor",
    DISAGREEMENT_ONLY: "Solo discrepancia",
    NO_VALUE: "Sin valor",
}

# Bins for the calibration curve. Wide enough that each carries enough matches
# for its observed rate to mean something on one season of data.
CALIBRATION_BINS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)

# The half-life the Valor tab fits with. Named rather than repeated, because the
# uncertainty band has to be measured on the *same* fit it describes -- a band
# from a differently-weighted model is an interval around a number that is not
# on screen.
FOOTBALL_LEDGER_PATH = "football_bets.csv"

VALUE_TAB_HALF_LIFE = 180


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
        # `opening_noise` writes the opening columns as well, the way a real
        # 2019/20-or-later file does. The closing source still wins resolution,
        # so every other surface sees exactly the frame it saw before; what it
        # buys is a CLV panel that has something to show on demo data.
        matches = preprocess_matches(
            generate_matches(seed=0, opening_noise=1.2).drop(
                columns=["TrueH", "TrueD", "TrueA"]), validate=False)
        is_demo = True

    return matches, is_demo, check_match_format(matches)


@st.cache_data(show_spinner=False)
def load_raw_frames(uploaded_bytes, paths):
    """The season files as published, before `processor.py` resolves one source.

    Every other surface wants the resolved frame; closing line value is the one
    question that needs both column families at once, and resolution is exactly
    what throws the second one away. Returns None when there is nothing to read.
    """
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))
    if paths:
        frames = [pd.read_csv(path) for path in paths]
        return pd.concat(frames, ignore_index=True) if frames else None
    return generate_matches(seed=0, opening_noise=1.2).drop(
        columns=["TrueH", "TrueD", "TrueA"])


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
def _run_backtest(cache_key, _matches, n_windows, half_life, method, models,
                  blend_weight, pool):
    """Walk-forward model-vs-market backtest over several models at once.

    `_matches` is underscore-prefixed so Streamlit does not try to hash the
    frame (it carries `.attrs`); identity comes from `cache_key` plus the
    parameters. Slow: it refits Dixon-Coles once per evaluated match, which is
    also why every model is scored in that one pass rather than one run each.
    """
    from football.backtest import compare_models

    # Floor on the training set; window_bounds already takes max(min_train,
    # n - n_windows), so this is just "never fit on fewer than 100 matches".
    return compare_models(_matches, n_windows=n_windows, min_train=100,
                          half_life=half_life, method=method, models=models,
                          blend_weight=blend_weight, pool=pool)


@st.cache_resource(show_spinner="Ajustando el Elo…")
def _fit_elo(_matches, cache_key):
    """Elo ratings, cached on the frame's fingerprint like the Dixon-Coles fit."""
    from football.elo import Elo

    return Elo.fit(_matches)


def _fingerprint(matches):
    """A cheap, hashable identity for a match frame Streamlit cannot hash itself."""
    return (
        len(matches),
        str(matches["ds"].min()),
        str(matches["ds"].max()),
        int(matches["home_goals"].sum()),
        int(matches["away_goals"].sum()),
    )


def _render_resolution(table):
    """What this backtest could have detected, beside what it did.

    The rule the lottery page has followed for a long time, arriving in
    football: a null result is unreadable without its resolution. "No model
    beat the closing line" over 40 matches and over 4,000 are the same sentence
    and completely different findings, and nothing in the table's shape says
    which one is on screen.

    The spread is taken from **this run's own forecasts**, never from a
    reference constant. Unlike the lottery's hypergeometric null there is no
    closed form for it: it depends on the league, the book and how far the
    model strays from the price, and a minimum detectable edge quoted from an
    assumed spread is a guess with a decimal point on it.
    """
    from football.power import minimum_detectable_edge, observed_score_sd

    forecasts = table.attrs.get("forecasts")
    if not forecasts or not forecasts["models"]:
        return

    rows = []
    for name, probs in forecasts["models"].items():
        spread = observed_score_sd(probs, forecasts["market"], forecasts["outcomes"])
        if not np.isfinite(spread) or spread <= 0:
            continue
        n = len(forecasts["outcomes"])
        rows.append({
            "Modelo": MODEL_ES.get(name, name),
            "Partidos": n,
            "Ventaja medida": float(table.loc[table["model"] == name, "effect"].iloc[0]),
            "Ventaja mínima detectable": minimum_detectable_edge(n, score_sd=spread)["absolute"],
        })
    if not rows:
        return

    st.markdown("**¿Qué habría podido ver este backtest?**", help=HELP["fb_resolution"])
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Ventaja medida": "{:+.4f}", "Ventaja mínima detectable": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "La última columna es la mejora de RPS más pequeña que esta cantidad de partidos "
        "habría detectado 8 veces de cada 10. Si la ventaja medida es menor que ella, «no le "
        "gana al mercado» habla del tamaño de la muestra y no del modelo. Un modelo bueno le "
        "saca a la cuota de cierre unas milésimas, y una temporada de una liga no las resuelve "
        "— por eso existe **¿Se movió el precio hacia ti?** en la pestaña de Valor, que sí "
        "converge en cientos de apuestas.",
        help=HELP["fb_resolution"],
    )


def _render_out_of_sample_calibration(table):
    """Calibration of the backtest's own forecasts — the out-of-sample version.

    Fed from `table.attrs["forecasts"]`, which `compare_models` attaches, so
    nothing is refitted to ask this second question of the same run.

    It sits **after** the verdict and says so in its own copy, because the one
    way to misread it is as a result. A forecast that simply copies the closing
    price is perfectly calibrated and has no edge whatsoever; calibration says
    whether the numbers mean what they claim, not whether they beat anything.
    """
    forecasts = table.attrs.get("forecasts")
    if not forecasts or not forecasts["models"]:
        return

    section("¿Sus porcentajes significan lo que dicen?", "fb_oos_calibration")
    outcomes = forecasts["outcomes"]
    series = {MODEL_ES.get(name, name): probs for name, probs in forecasts["models"].items()}
    series["Mercado"] = forecasts["market"]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                name="Calibración perfecta", line=dict(dash="dash")))
    rows = []
    for label, probs in series.items():
        curve = reliability_curve(probs, outcomes)
        usable = [r for r in curve if not np.isnan(r["observed_frequency"])]
        if usable:
            figure.add_trace(go.Scatter(
                x=[r["mean_forecast"] for r in usable],
                y=[r["observed_frequency"] for r in usable],
                mode="markers+lines", name=label,
                marker=dict(size=[min(6 + r["count"] / 12, 22) for r in usable]),
                text=[f"{r['count']} pronósticos" for r in usable]))
        report = expected_calibration_error(probs, outcomes)
        rows.append({"Serie": label, "Error de calibración": report["ece"],
                     "Cubre": report["coverage"], "Grupos usados": report["n_bins_used"]})

    figure.update_layout(xaxis_title="Lo que dijo el modelo",
                         yaxis_title="Lo que pasó de verdad", height=380,
                         xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    chart(figure, "Curva de calibración fuera de muestra", "fb_oos_reliability")

    st.dataframe(pd.DataFrame(rows).style.format(
        {"Error de calibración": "{:.4f}", "Cubre": "{:.0%}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "«Cubre» es la parte de los pronósticos que cayó en grupos con partidos suficientes para "
        "medirlos. Un error pequeño sobre una décima parte de los pronósticos no es un error "
        "pequeño.",
        help=HELP["fb_oos_calibration"],
    )

    st.markdown("**¿Pronostica cada resultado tan seguido como pasa?**", help=HELP["fb_citl"])
    model_label, model_probs = next(iter(series.items()))
    citl = pd.DataFrame(calibration_in_the_large(model_probs, outcomes))
    citl["Resultado"] = citl["outcome"].map(OUTCOME_ES)
    citl["¿Desviado? (corregido)"] = citl["miscalibrated_corrected"].map(
        {True: "Sí", False: "No"})
    st.dataframe(
        citl[["Resultado", "mean_forecast", "base_rate", "difference", "p_value",
              "¿Desviado? (corregido)"]]
        .rename(columns={"mean_forecast": f"Media de {model_label}",
                         "base_rate": "Frecuencia real", "difference": "Diferencia",
                         "p_value": "p (dos colas)"})
        .style.format({f"Media de {model_label}": "{:.1%}", "Frecuencia real": "{:.1%}",
                       "Diferencia": "{:+.1%}", "p (dos colas)": "{:.3f}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "Son tres pruebas sobre los mismos partidos, así que la columna corregida es la que "
        "cuenta. Aquí la prueba es de **dos colas** a propósito: pronosticar de más y "
        "pronosticar de menos son los dos un error de calibración.",
    )


def render():
    st.title("⚽ Fútbol")
    st.caption(
        "La barra que hay que superar aquí es la **cuota de cierre** — el precio después de que se "
        "movió todo el dinero — no un Elo ni un 50/50. Los modelos se muestran siempre junto al "
        "mercado; el veredicto medido, fuera de muestra y corregido, está en "
        "**4 · ¿Le gana al mercado?**, y nada de **5 · Valor** significa algo hasta que lo corras. "
        "Si una palabra no te suena, está en el **Glosario** de la barra lateral."
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
    glossary()

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
        # Colombia's extra files are opening prices only, so there is no second
        # end of the line to measure against. The panel says so rather than
        # silently not appearing.
        raw_frame = None
    else:
        paths = tuple(os.path.join(directory, name) for name in chosen) if not uploaded else ()
        raw_frame = load_raw_frames(uploaded.getvalue() if uploaded else None, paths)

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

    tabs = st.tabs(["1 · Datos", "2 · Mercado", "3 · Pronóstico", "4 · ¿Le gana al mercado?",
                    "5 · Valor", "6 · Registro"])

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
                model_cache_key = (*_fingerprint(matches), 0)
                try:
                    model = _fit_dixon_coles(matches, model_cache_key)
                    model_probs = model.predict_matches(priced)
                    # The binning rule lives in football/calibration.py so that
                    # this diagnostic and the out-of-sample curve in tab 4 drop
                    # a too-sparse bin by the same standard.
                    curve = reliability_curve(model_probs, list(priced["outcome"]),
                                              bins=len(CALIBRATION_BINS) - 1, outcome="H")
                    m_summary = pd.DataFrame([
                        {"bucket": f"[{r['bin_low']:.1f}, {r['bin_high']:.1f})",
                         "predicha": r["mean_forecast"], "observada": r["observed_frequency"],
                         "partidos": r["count"]}
                        for r in curve if r["count"]])

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
                        "Ajustado sobre las mismas temporadas que se muestran (dentro de "
                        "muestra), así que esto favorece al modelo: la versión **fuera de "
                        "muestra**, sobre partidos que el modelo no vio, está en "
                        "**4 · ¿Le gana al mercado?**, debajo del veredicto. Los grupos con "
                        "muy pocos partidos salen vacíos a propósito: una frecuencia sobre tres "
                        "partidos no es una medición."
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
                for column, outcome in zip(PROBABILITY_COLUMNS, OUTCOMES, strict=True)}})
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
        for column, outcome, share in zip(columns, OUTCOMES, shares, strict=True):
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
            c1, c2 = st.columns(2)
            n_windows = c1.slider("Partidos a evaluar (walk-forward)", 20, 200, 40, step=10)
            half_life = c2.number_input(
                "Vida media (días), 0 = sin decaimiento",
                min_value=0, value=180, step=30, help=HELP["fb_half_life"])
            chosen_models = st.multiselect(
                "Modelos a medir", list(MODEL_NAMES), default=list(MODEL_NAMES),
                format_func=lambda name: MODEL_ES[name], help=HELP["fb_models_pick"])

            b1, b2 = st.columns(2)
            blend_weight = b1.slider("Peso del modelo en la mezcla", 0.0, 1.0, 0.5, step=0.05,
                                     help=HELP["fb_blend_weight"])
            pool = b2.selectbox("Regla de mezcla", list(POOLS), help=HELP["fb_pool"])
            st.caption(
                "Con peso 0 la mezcla **es** el mercado y puntúa exactamente igual, así que "
                "cualquier mejora al subir el peso es información que el modelo tiene y el precio "
                "no. Esa es una pregunta más útil que «¿le gana al mercado?», que casi ningún "
                "modelo responde que sí.",
                help=HELP["fb_blend"],
            )

            eval_key = (*_fingerprint(matches), source, n_windows, int(half_life), method,
                        tuple(chosen_models), blend_weight, pool)
            if st.button("Correr backtest") and chosen_models:
                with st.spinner("Reajustando los modelos por ventana…"):
                    table = _run_backtest(
                        eval_key, matches, n_windows, int(half_life) or None, method,
                        tuple(chosen_models), blend_weight, pool)
                st.session_state["fb_backtest"] = table

            table = st.session_state.get("fb_backtest")
            if table is not None:
                winners = [MODEL_ES[m] for m in
                           table.loc[table["beats_market_corrected"], "model"]]
                best = table.loc[table["skill_score"].idxmax()]
                plain_verdict(
                    bool(winners),
                    f"{', '.join(winners)} superó a la cuota de cierre"
                    if winners else
                    "Ningún modelo superó a la cuota de cierre",
                    f"El mejor puntuó {best['model_score']:.4f} de RPS contra "
                    f"{best['market_score']:.4f} del mercado, sobre "
                    f"{int(best['n_windows_scored'])} partidos "
                    f"(ventaja {best['effect']:+.4f}, IC 95% "
                    f"[{best['ci_low']:+.4f}, {best['ci_high']:+.4f}]). "
                    + ("Antes de creerlo, cámbiale el método de de-margen en la barra lateral: si "
                       "la conclusión se cae, el hallazgo era sobre el margen."
                       if winners else
                       "Es el resultado esperado: la cuota de cierre es el precio después de que "
                       "se movió todo el dinero, y casi nada le gana."),
                )
                _render_resolution(table)

                display = table.copy()
                display["Modelo"] = display["model"].map(MODEL_ES)
                display["Ventaja (IC 95%)"] = display.apply(
                    lambda r: f"{r['effect']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                    axis=1)
                display["¿Gana? (naive)"] = display["beats_market"].map({True: "Sí", False: "No"})
                display["¿Gana? (corregido)"] = display["beats_market_corrected"].map(
                    {True: "Sí", False: "No"})
                st.markdown("**Veredicto por modelo**", help=HELP["fb_models_table"])
                st.dataframe(
                    display[["Modelo", "n_windows_scored", "model_score", "market_score",
                             "skill_score", "Ventaja (IC 95%)", "p_value_greater",
                             "bonferroni_threshold", "¿Gana? (naive)", "¿Gana? (corregido)"]]
                    .rename(columns={"n_windows_scored": "Partidos", "model_score": "RPS modelo",
                                     "market_score": "RPS mercado", "skill_score": "Skill",
                                     "p_value_greater": "p (una cola)",
                                     "bonferroni_threshold": "Umbral corregido"})
                    .style.format({"RPS modelo": "{:.4f}", "RPS mercado": "{:.4f}",
                                   "Skill": "{:+.4f}", "p (una cola)": "{:.4f}",
                                   "Umbral corregido": "{:.4f}"}),
                    use_container_width=True, hide_index=True)

                figure = go.Figure()
                figure.add_bar(x=[MODEL_ES[m] for m in table["model"]], y=table["model_score"],
                               name="Modelo")
                figure.add_hline(y=float(table["market_score"].iloc[0]), line_dash="dash",
                                 line_color="gray", annotation_text="Mercado")
                figure.update_layout(yaxis_title="RPS (menor es mejor)", height=340)
                chart(figure, "Puntuación de cada modelo contra el mercado", "fb_models_table")

                st.caption(
                    f"El umbral corregido es 0.05 dividido entre {len(table)} porque se midieron "
                    f"{len(table)} modelos contra los mismos partidos. Mira siempre esa columna: "
                    "un skill score positivo con intervalo que cruza 0 no es una ventaja, es ruido "
                    "con el signo favorable.",
                    help=HELP["fb_beats_market"],
                )

                _render_out_of_sample_calibration(table)

    # ------------------------------------------------------------------ Valor
    # Rendered after the backtest block, not beside the forecast: Streamlit runs
    # every tab body on each rerun in source order, so a Valor placed earlier
    # would read the verdict out of session state one rerun stale and tell the
    # reader nothing had been measured on the very click that measured it.
    with tabs[4]:
        render_value_tab(matches, method)
        _render_bankroll(matches, method)
        _render_clv(raw_frame, method, is_demo)

    # --------------------------------------------------------------- Registro
    with tabs[5]:
        render_registry_tab(matches, method, is_demo)


def render_registry_tab(matches, method, is_demo):
    """A forward record for football: the forecast first, then what was staked on it.

    Two surfaces that must not merge. The **registry** stores a forecast before
    kick-off and scores it against the de-margined closing price — that is the
    domain's bar and the only thing that can say whether the model is worth
    anything. The **ledger** stores money. A row that was both would be read as
    whichever of the two suits the reader, so they are separate files, separate
    modules and separate blocks on this page.

    Recording goes through `football/registry.py`, which refuses a fixture that
    has already kicked off and a second forecast under the same label. The
    dashboard is another caller of those refusals, not a way round them.
    """
    from football.registry import load as load_registry
    from football.registry import record as record_forecast
    from football.registry import score_pending as score_registry
    from football.registry import summary as registry_summary

    section("Pronósticos registrados antes del partido", "fb_tab_registro")
    if is_demo:
        st.warning(
            "Estás sobre partidos sintéticos. Puedes registrar, pero se puntuará contra "
            "resultados inventados. Carga temporadas reales antes de empezar un registro que "
            "quieras tomar en serio."
        )

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    if len(teams) < 2:
        st.info("Hacen falta al menos dos equipos en los datos cargados.")
        return

    c1, c2 = st.columns(2)
    home = c1.selectbox("Local", teams, key="reg_home")
    away = c2.selectbox("Visitante", [t for t in teams if t != home], key="reg_away")

    st.markdown("**Las cuotas que te dan ahora mismo**", help=HELP["fb_registry_odds"])
    o1, o2, o3 = st.columns(3)
    odds = (o1.number_input("Local", min_value=1.01, value=2.10, step=0.05, key="reg_odds_h"),
            o2.number_input("Empate", min_value=1.01, value=3.40, step=0.05, key="reg_odds_d"),
            o3.number_input("Visitante", min_value=1.01, value=3.60, step=0.05, key="reg_odds_a"))

    try:
        model = _fit_dixon_coles(matches, (len(matches), 0)).predict_outcome(home, away)
    except UnknownTeamError as exc:
        st.warning(f"El modelo no conoce a uno de los dos equipos: {exc}")
        return

    market = implied_probabilities(odds, method=method)
    # The rule this page never breaks: model probabilities are shown only beside
    # the de-margined price, because a bare model column implies it is good on
    # its own and that is the one thing no surface here may suggest.
    st.dataframe(
        pd.DataFrame({
            "Resultado": ["Local", "Empate", "Visitante"],
            "Modelo": model, "Mercado (sin margen)": market,
            "Cuota cruda": odds,
        }).style.format({"Modelo": "{:.1%}", "Mercado (sin margen)": "{:.1%}",
                         "Cuota cruda": "{:.2f}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        f"Margen del libro: **{overround(odds):.1%}**. El modelo se mide contra la columna "
        "sin margen; una apuesta, contra la cuota cruda. Son dos varas distintas y el margen está "
        "en medio.", help=HELP["fb_registry_odds"])

    r1, r2, r3 = st.columns(3)
    match_date = r1.date_input("Fecha del partido", key="reg_match_date")
    label = r2.text_input("Etiqueta", "dixon-coles", key="reg_fb_label")
    note = r3.text_input("Nota (opcional)", "", key="reg_fb_note")

    if st.button("Registrar pronóstico"):
        try:
            row = record_forecast(model, match_date, home, away,
                                  label.strip() or "sin etiqueta", note=note)
        except Exception as exc:  # noqa: BLE001 — the guard's message is the content
            st.error(
                "**No se registró.** El registro rechaza un partido que ya se jugó y un segundo "
                "pronóstico con la misma etiqueta: cada fila tiene que haber sido falsable cuando "
                "se escribió, y nada se sobreescribe."
            )
            st.caption(f"Detalle: {exc}")
        else:
            st.success(
                f"Registrado {row['home_team']} vs {row['away_team']} para el "
                f"{pd.to_datetime(row['match_date']):%Y-%m-%d} como `{row['label']}`."
            )
            st.rerun()

    st.divider()
    registry = load_registry()
    m1, m2 = st.columns(2)
    m1.metric("Pronósticos registrados", len(registry))
    m2.metric("Puntuados", int(registry["model_score"].notna().sum()) if len(registry) else 0)

    if st.button("Puntuar los que ya se jugaron", help=HELP["fb_registry_score"]):
        scored = matches.copy()
        probabilities = market_probabilities(scored, method=method)
        for column, values in probabilities.items():
            scored[column] = values
        score_registry(scored)
        st.rerun()

    table = registry_summary()
    if not table.empty:
        row = table.iloc[0]
        plain_verdict(
            bool(row["beats_market_corrected"]),
            ("Le gana al mercado en el registro, con la corrección aplicada."
             if row["beats_market_corrected"]
             else "No hay evidencia de que el registro le gane al mercado."),
            (f"n = {int(row['n_scored'])} · p (una cola) = {row['p_value_greater']:.3f} · "
             f"diferencia media de RPS = {row['effect']:+.4f}"),
        )
        st.caption(
            "Esta es la vara de este dominio: el precio de cierre sin margen sobre los mismos "
            "partidos. Un pronóstico que **es** el mercado acumula exactamente cero, y por eso "
            "cualquier cifra distinta de cero significa algo.", help=HELP["fb_registry_verdict"])
    elif len(registry):
        st.info("Nada puntuado todavía: ningún partido registrado se ha jugado (o falta su precio).")

    st.divider()
    betlog_page.render(
        FOOTBALL_LEDGER_PATH,
        "Lo que apostaste",
        HELP["fb_registry_selection"],
        default_label="dixon-coles",
        selection_options=[f"{home} (local)", "Empate", f"{away} (visitante)"],
    )


@st.cache_data(show_spinner="Midiendo la incertidumbre del modelo…")
def _bootstrap_bands(cache_key, _matches, home_team, away_team, half_life, n_resamples):
    """Bootstrap band for one fixture, cached on the frame's fingerprint.

    Underscore-prefixed frame for the reason every cached helper here uses one:
    Streamlit cannot hash a frame carrying `.attrs`.
    """
    from football.dixon_coles import DixonColes
    from football.uncertainty import bootstrap_predictions

    return bootstrap_predictions(
        _matches, [(home_team, away_team)],
        lambda data, hl: DixonColes.fit(data, half_life=hl),
        n_resamples=n_resamples, half_life=half_life)


def _with_uncertainty(table, matches, home_team, away_team, half_life, n_resamples=120):
    """Attach the band to a value table and demote what it does not support.

    Never upgrades. A band whose top end clears the price while the point does
    not is still a model without an edge on its own estimate, and promoting it
    would turn an interval into a second opinion.
    """
    from football.uncertainty import with_bands

    try:
        bands = _bootstrap_bands((*_fingerprint(matches), half_life), matches,
                                 home_team, away_team, half_life, n_resamples)
    except Exception as exc:  # noqa: BLE001 — a band is an extra, never the blocker
        st.caption(f"No se pudo medir la incertidumbre del modelo: {exc}")
        return table

    banded = with_bands(table, bands)
    demoted = int((banded["verdict"] != banded["verdict_point"]).sum())
    if demoted:
        st.warning(
            f"**{demoted} resultado(s) dejaron de contar como apuesta al mirar el intervalo.** "
            "El modelo supera el precio con su estimación puntual, pero el intervalo de esa "
            "estimación no lo supera — es decir, la ventaja podría ser ruido con el signo "
            "favorable. Se degradan a «solo discrepancia», nunca al revés."
        )
    return banded


def _render_bankroll(matches, method):
    """What running these bets would have felt like, with the null drawn beside it.

    Gated behind the measured verdict, exactly as the staking table above is.
    A bankroll curve is the most persuasive object this project can produce and
    the easiest to mislead with, so it does not appear at all until the model
    has been measured, and never appears without the companion curve showing
    what the same bets do when the edge is not real.
    """
    from football.bankroll import DEFAULT_PATHS, simulate_bankroll, stake_fraction_sweep, summarise

    verdict = st.session_state.get("fb_backtest")
    forecasts = None if verdict is None else verdict.attrs.get("forecasts")
    if not forecasts or not forecasts["models"]:
        return

    section("¿Cómo se sentiría correr estas apuestas?", "fb_bankroll")

    # The forecasts come from the backtest's held-out matches; the raw prices
    # have to be looked up for the same ones, because a bet pays the margin.
    outcomes = forecasts["outcomes"]
    priced = matches.dropna(subset=list(ODDS_COLUMNS)).tail(len(outcomes))
    if len(priced) != len(outcomes):
        st.info("No hay cuotas crudas para todos los partidos evaluados; no se puede simular.")
        return

    name = next(iter(forecasts["models"]))
    model_probs = forecasts["models"][name]
    odds = priced[list(ODDS_COLUMNS)].to_numpy(dtype=float)

    fraction = st.slider("Fracción de Kelly", 0.05, 1.0, 0.25, step=0.05,
                         help=HELP["fb_kelly_fraction"])
    simulation = simulate_bankroll(model_probs, forecasts["market"], odds, outcomes,
                                   fraction=fraction, n_paths=DEFAULT_PATHS, seed=0)
    if not simulation["n_staked"]:
        st.info(
            f"**{MODEL_ES.get(name, name)}** no encontró ninguna apuesta con valor en estos "
            "partidos: su probabilidad nunca superó a la cuota cruda. Es el resultado más "
            "común y no es un error."
        )
        return

    figure = go.Figure()
    for label, paths, colour in (("Si la ventaja es real", simulation["paths"], None),
                                 ("Si no lo es", simulation["null_paths"], "#999999")):
        median = np.median(paths, axis=0)
        figure.add_trace(go.Scatter(y=median, mode="lines", name=label,
                                    line=dict(color=colour) if colour else None))
        figure.add_trace(go.Scatter(y=np.quantile(paths, 0.9, axis=0), mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(y=np.quantile(paths, 0.1, axis=0), mode="lines",
                                    line=dict(width=0), fill="tonexty", opacity=0.2,
                                    showlegend=False, hoverinfo="skip"))
    figure.update_layout(xaxis_title="Apuestas", yaxis_title="Banca (empieza en 1)",
                         yaxis_type="log", height=360)
    chart(figure, "Banca simulada, con y sin ventaja real", "fb_bankroll_paths")

    st.dataframe(summarise(simulation).style.format({
        "median_roi": "{:+.1%}", "roi_ci_low": "{:+.1%}", "roi_ci_high": "{:+.1%}",
        "share_losing": "{:.0%}", "median_drawdown": "{:.0%}",
        "worst_drawdown_95": "{:.0%}", "risk_of_ruin": "{:.0%}"}),
        use_container_width=True, hide_index=True)

    st.markdown("**El precio de apostar más fuerte**", help=HELP["fb_kelly_fraction"])
    sweep = stake_fraction_sweep(model_probs, forecasts["market"], odds, outcomes,
                                 n_paths=200, seed=0)
    st.dataframe(sweep.style.format({
        "median_roi": "{:+.1%}", "median_drawdown": "{:.0%}",
        "worst_drawdown_95": "{:.0%}", "risk_of_ruin": "{:.0%}",
        "share_losing": "{:.0%}"}), use_container_width=True, hide_index=True)
    st.caption(
        "La caída máxima crece mucho más rápido que el retorno. Kelly entero maximiza el "
        "crecimiento **si la probabilidad es correcta**, y la de un modelo es una estimación "
        "con error — por eso el valor por defecto es un cuarto. En datos sintéticos, un modelo "
        "al que se le da la verdad exacta se arruina igual apostando a Kelly entero.",
        help=HELP["fb_bankroll"],
    )


def _render_clv(raw_frame, method, is_demo=False):
    """Closing line value: did the price move toward the bet after it was taken?

    Rendered inside **Valor** and after the staking block, because it belongs
    to the same question and answers a narrower version of it. "Does this model
    beat the closing line" needs thousands of matches to answer; "did the line
    move toward me" needs hundreds, because it is a direct measurement rather
    than a difference of two noisy scores.

    The three fixed strategies are the point of the table, not filler. Backing
    every home side involves no selection at all, so its CLV is a fact about how
    this book's line drifts and nothing about anyone's skill. They are the
    control the model's row is read against.
    """
    from football.clv import PriceJoinError, beats_closing_test, clv_table, paired_prices

    section("¿Se movió el precio hacia ti?", "fb_clv")
    if raw_frame is None:
        st.info(
            "Los archivos «extra» (Colombia) traen **solo cuotas de apertura**, así que no hay "
            "un segundo extremo de la línea contra el cual medir. Esto necesita un archivo "
            "europeo de 2019/20 en adelante, donde las columnas de apertura y de cierre vienen "
            "juntas."
        )
        return

    try:
        joined = paired_prices(raw_frame)
    except (PriceJoinError, ValueError) as exc:
        st.info(
            "Estos datos no traen los dos extremos de la línea. football-data publica cuotas de "
            "cierre solo desde 2019/20; antes de eso lo mejor disponible es la apertura, y medir "
            "una apertura contra sí misma no es valor de línea de cierre."
        )
        st.caption(f"Detalle: {exc}")
        return

    strategies = {OUTCOME_ES[outcome]: [outcome] * len(joined) for outcome in OUTCOMES}
    rows = []
    for label, bets in strategies.items():
        table = clv_table(joined, bets=bets, method=method)
        result = beats_closing_test(table["clv"], n_comparisons=len(strategies))
        rows.append({"Apuesta": label, "Partidos": result["n_bets"],
                     "Valor medio": result["mean_clv"],
                     "% a favor": result["hit_rate"],
                     "¿Gana al cierre? (corregido)":
                         "Sí" if result["beats_closing_corrected"] else "No"})

    st.dataframe(pd.DataFrame(rows).style.format(
        {"Valor medio": "{:+.4f}", "% a favor": "{:.0%}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        f"Medido sobre {len(joined)} partidos, de la cuota de apertura "
        f"(**{joined.attrs['bet_odds_source']}**) a la de cierre "
        f"(**{joined.attrs['closing_odds_source']}**), con el margen quitado de los dos lados. "
        "Estas tres son estrategias sin ninguna selección: apostar siempre al local no requiere "
        "saber nada, así que su valor dice cómo se mueve la línea de esta casa y nada sobre "
        "nadie. Son el control contra el que se leería la fila de un modelo.",
        help=HELP["fb_clv"],
    )
    if is_demo:
        st.warning(
            "Son **datos sintéticos**. Si alguna fila sale «Sí», es una propiedad del generador "
            "—la apertura se simula como una versión borrosa del cierre— y no una estrategia. "
            "Carga una temporada real para que esta tabla diga algo."
        )


def render_value_tab(matches, method):
    """Edge and staking, with the margin band made visible and the verdict gated.

    The one surface here that can lose someone money. Two things hold it
    together: the stake is never shown without the measured verdict beside it,
    and the gap between the two bars — what the market thinks versus what you
    have to beat — is drawn rather than described, because a model that is more
    optimistic than the price and still short of `1/odds` looks exactly like a
    bet until you see the gap.
    """
    from football.value import margin_cost, value_table

    section("¿Hay valor en esta cuota?", "fb_tab_valor")

    verdict = st.session_state.get("fb_backtest")
    if verdict is None:
        st.warning(
            "**Todavía no has medido el modelo.** Corre el backtest en "
            "**4 · ¿Le gana al mercado?** primero. Todo lo de esta pestaña es condicional a que "
            "el modelo sea bueno, y sin esa medida no hay nada que lo respalde."
        )
    else:
        winners = [MODEL_ES[m] for m in verdict.loc[verdict["beats_market_corrected"], "model"]]
        plain_verdict(
            bool(winners),
            f"Medido: {', '.join(winners)} superó al mercado" if winners else
            "Medido: ningún modelo superó al mercado",
            "Las apuestas de abajo solo tienen sentido si esa línea dice que sí. Si dice que no, "
            "léelas como un ejercicio: el modelo discrepa del precio y no hay evidencia de que "
            "tenga razón.",
        )

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    if len(teams) < 2 or len(matches) < 150:
        st.info("Hacen falta al menos dos equipos y ~150 partidos para ajustar el modelo.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    home_team = c1.selectbox("Local", teams, index=0, key="value_home")
    away_team = c2.selectbox("Visitante", teams, index=1, key="value_away")
    bankroll = c3.number_input("Banca (COP)", min_value=0, value=1_000_000, step=100_000)
    if home_team == away_team:
        st.warning("Elige dos equipos distintos.")
        return

    o1, o2, o3 = st.columns(3)
    odds = (
        o1.number_input("Cuota local", min_value=1.01, value=2.10, step=0.05, key="v_home"),
        o2.number_input("Cuota empate", min_value=1.01, value=3.40, step=0.05, key="v_draw"),
        o3.number_input("Cuota visitante", min_value=1.01, value=3.80, step=0.05, key="v_away"),
    )

    try:
        model = _fit_dixon_coles(matches, (*_fingerprint(matches), VALUE_TAB_HALF_LIFE))
        model_probabilities = model.predict_outcome(home_team, away_team)
    except UnknownTeamError as exc:
        st.error(f"El modelo no conoce a ese equipo en las temporadas cargadas. Detalle: {exc}")
        return

    table = value_table(model_probabilities, np.array(odds), method=method,
                        fraction=DEFAULT_KELLY_FRACTION, bankroll=bankroll)

    st.markdown("**Las dos barras, que no son la misma**", help=HELP["fb_two_bars"])
    figure = go.Figure()
    labels = [OUTCOME_ES[o] for o in OUTCOMES]
    figure.add_bar(x=labels, y=table["market_probability"], name="Lo que cree el mercado")
    figure.add_bar(x=labels, y=table["break_even_probability"] - table["market_probability"],
                   name="Margen de la casa", marker_color="#d9b38c")
    figure.add_scatter(x=labels, y=table["model_probability"], mode="markers", name="El modelo",
                       marker=dict(size=16, symbol="diamond", color="#d62728"))
    figure.update_layout(barmode="stack", yaxis_tickformat=".0%",
                         yaxis_title="Probabilidad", height=360)
    chart(figure, "Modelo, mercado y lo que hay que superar", "fb_margin_cost")
    st.caption(
        "El rombo rojo es el modelo. La barra azul es lo que el mercado cree de verdad; la franja "
        "de arriba es su comisión. **Hay apuesta solo si el rombo queda por encima de toda la "
        "columna** — si cae dentro de la franja, el modelo discrepa del mercado pero no lo "
        f"suficiente para pagar el margen, que aquí vale {margin_cost(np.array(odds), method=method).mean():.1%} "
        "de probabilidad por resultado."
    )

    # The band, and the downgrade it forces. A point clearing the price by a
    # hair while its interval straddles it is noise with a favourable sign, and
    # showing a stake on it is the one thing this tab must not do.
    table = _with_uncertainty(table, matches, home_team, away_team, VALUE_TAB_HALF_LIFE)

    display = table.copy()
    display["Resultado"] = [OUTCOME_ES[o] for o in display["outcome"]]
    display["Estado"] = display["verdict"].map(VERDICT_ES)
    st.markdown("**Detalle por resultado**", help=HELP["fb_value_table"])
    st.dataframe(
        display[["Resultado", "odds", "model_probability", "market_probability",
                 "break_even_probability", "expected_value", "kelly_stake", "stake", "Estado"]]
        .rename(columns={"odds": "Cuota", "model_probability": "Modelo",
                         "market_probability": "Mercado (sin margen)",
                         "break_even_probability": "Hay que superar",
                         "expected_value": "Valor esperado", "kelly_stake": "% de la banca",
                         "stake": "Apuesta (COP)"})
        .style.format({"Cuota": "{:.2f}", "Modelo": "{:.1%}", "Mercado (sin margen)": "{:.1%}",
                       "Hay que superar": "{:.1%}", "Valor esperado": "{:+.3f}",
                       "% de la banca": "{:.2%}", "Apuesta (COP)": "${:,.0f}"}),
        use_container_width=True, hide_index=True)

    staked = table[table["kelly_stake"] > 0]
    if staked.empty:
        st.info(
            "**Ninguna apuesta a estas cuotas.** El modelo no supera el precio en ningún "
            "resultado, que es lo normal: la mayoría de los partidos no tienen valor para "
            "ningún modelo, y un sistema que siempre encuentra una apuesta está encontrando "
            "el margen de la casa."
        )
    else:
        st.caption(
            "Las cifras de apuesta son **un cuarto de Kelly**, no Kelly entero. Kelly es la "
            "apuesta óptima suponiendo que tu probabilidad es correcta; la de un modelo es una "
            "estimación con error, y sobre una ventaja que no existe Kelly sube la apuesta justo "
            "cuando el modelo está más seguro y más equivocado.",
            help=HELP["fb_kelly"],
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
    for col, label, value in zip((m1, m2, m3), ("Local", "Empate", "Visitante"), p_model, strict=True):
        col.metric(label, f"{value:.1%}")
    if not all(o > 1.0 for o in (odd_home, odd_draw, odd_away)):
        st.caption(
            "Estas cifras son solo el modelo: no hay cuotas para este partido, así que no hay "
            "mercado contra el que contrastarlas."
        )

    # --- the cheap baseline, beside the expensive model ---
    try:
        elo = _fit_elo(matches, _fingerprint(matches))
        p_elo = elo.predict_outcome(home_team, away_team)
    except UnknownTeamError:
        p_elo = None

    if p_elo is not None:
        st.markdown("**Y lo que dice el Elo**", help=HELP["fb_elo"])
        e1, e2, e3 = st.columns(3)
        for col, label, value in zip((e1, e2, e3), ("Local", "Empate", "Visitante"), p_elo, strict=True):
            col.metric(label, f"{value:.1%}")
        st.caption(
            f"Elo de {home_team}: **{elo.rating(home_team):.0f}** · "
            f"{away_team}: **{elo.rating(away_team):.0f}** "
            f"(diferencia con la ventaja de local incluida: "
            f"{elo.rating_gap(home_team, away_team):+.0f}). "
            "Si Dixon-Coles no se separa del Elo, no está aportando nada sobre una sola nota de "
            "fuerza por equipo — y el Elo es mucho más barato. Cuál gana de verdad está en "
            "**4 · ¿Le gana al mercado?**",
            help=HELP["fb_elo_ranking"],
        )
        with st.expander("Ver la tabla de fuerza Elo completa"):
            ranking = pd.DataFrame(elo.ranking(), columns=["Equipo", "Elo"])
            ranking.insert(0, "#", range(1, len(ranking) + 1))
            st.dataframe(ranking.style.format({"Elo": "{:.0f}"}),
                         use_container_width=True, hide_index=True)

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
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_model, name="Dixon-Coles"))
        if p_elo is not None:
            figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_elo, name="Elo"))
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_market, name="Mercado"))
        figure.update_layout(barmode="group", yaxis_tickformat=".0%", height=340)
        chart(figure, "Probabilidades: modelos y mercado", "fb_model_vs_market")
        comparison = {
            "Resultado": ["Local", "Empate", "Visitante"],
            "Dixon-Coles": [f"{p:.1%}" for p in p_model],
            "Mercado": [f"{p:.1%}" for p in p_market],
            "DC − Mercado": [f"{m - k:+.1%}" for m, k in zip(p_model, p_market, strict=True)],
        }
        if p_elo is not None:
            comparison["Elo"] = [f"{p:.1%}" for p in p_elo]
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
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
