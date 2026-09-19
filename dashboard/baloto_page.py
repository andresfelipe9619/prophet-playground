"""The Baloto page: the original dashboard, unchanged in substance.

Run through `dashboard/app.py`, which owns the sidebar's domain selector.

The goal here is decision support, not "the number to play": every tab that
touches a model or a heuristic ("hot numbers", "overdue numbers", forecasts) is
paired with the statistical check for whether that signal is distinguishable
from pure chance. Baloto draws are independent and uniform by design, so the
honest expectation is that most of these checks come back negative — this page
is built to show that clearly instead of hiding it.
"""

import importlib.util
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lottery.backtest as bt
from dashboard.ui import HELP, chart, glossary, plain_verdict, section, verdict_badge
from lottery.analysis.popularity import (
    compare_tickets,
    popularity_components,
    popularity_score,
)
from lottery.analysis.power import (
    describe as power_describe,
)
from lottery.analysis.power import (
    minimum_detectable_effect,
    power_curve,
    required_draws_table,
    super_minimum_detectable_effect,
)
from lottery.analysis.prizes import (
    breakeven_jackpot,
    category_probabilities,
    expected_value,
    total_combinations,
)
from lottery.analysis.randomness import (
    frequency_table,
    gap_table,
    hot_cold_numbers,
    is_sorted_ascending,
    pooled_uniformity_test,
    randomness_report,
)
from lottery.analysis.registry import (
    RegistryError,
    score_pending,
)
from lottery.analysis.registry import (
    load as load_registry,
)
from lottery.analysis.registry import (
    pending as pending_predictions,
)
from lottery.analysis.registry import (
    record as record_prediction,
)
from lottery.analysis.registry import (
    status as registry_status,
)
from lottery.analysis.registry import (
    summary as registry_summary,
)
from lottery.analysis.sensitivity import (
    DEFAULT_STRENGTHS,
    sensitivity_report,
    sensitivity_threshold,
)
from lottery.analysis.structure import (
    CALENDAR_THRESHOLD,
    structure_report,
    sum_percentile,
)
from lottery.analysis.tickets import (
    STRATEGIES,
    Ticket,
    check_against_history,
    check_ticket,
    compare_strategies,
    draw_from_row,
    generate_portfolio,
    history_summary,
    portfolio_coverage,
    stability_check,
)
from lottery.models.baseline import expected_main_matches, most_frequent_pick
from lottery.models.common import (
    DEFAULT_DATA_PATH,
    MAIN_BALL_RANGE,
    MAIN_BALLS_DRAWN,
    SUPER_BALL_RANGE,
    build_position_series,
    infer_draw_weekdays,
    main_positions,
    next_draw_dates,
    series_label,
    super_position,
)
from lottery.models.statsforecast_model import MODEL_NAMES, adjusted_predictions, fit_predict_all
from lottery.models.timesfm_model import CheckpointUnavailableError
from lottery.models.timesfm_model import forecast_positions as timesfm_forecast
from lottery.models.timesfm_model import is_available as timesfm_is_available
from lottery.models.xgboost_model import forecast_next
from lottery.utils.processor import (
    check_draw_format,
    current_format_mask,
    load_and_preprocess,
    preprocess_draws,
)
from lottery.utils.sample_data import load_sample_and_preprocess

MIN_TRAIN_FLOOR = 20  # below this the models have nothing to learn from

# Prophet ships in every environment this app is meant to run in, deployment
# included — a hosted instance missing a model is a different app, not a smaller
# one. This is the net for the case where the install is incomplete anyway: the
# option disappears and says why, instead of raising an ImportError from inside
# a spinner after the reader has already clicked the button. `find_spec` does
# not import the package, so the check costs nothing on the common path where it
# is present. See docs/local-setup.md.
PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None

# TimesFM ships in requirements.txt like everything else, because this project
# runs locally and there is no free-tier memory cap to design around. The check
# stays because the install is genuinely heavy (torch plus a downloaded
# checkpoint) and someone may well skip it deliberately — a missing model should
# say so rather than vanish. `is_available()` uses find_spec and does not import
# torch, which matters on a surface Streamlit re-executes top to bottom on every
# interaction. See docs/local-setup.md.
TIMESFM_AVAILABLE = timesfm_is_available()


@st.cache_data(show_spinner=False)
def load_data(path, uploaded_bytes, current_format_only):
    """Load draws and derive everything downstream needs, in one cached step.

    position_series comes back from here rather than from a second cached
    function so Streamlit never has to hash the full frames as arguments on
    every rerun. `format_report` travels with the data so the UI can say what
    was dropped and why — the filtering is invisible in the frames themselves.
    """
    if uploaded_bytes is not None:
        df, balls_expanded = preprocess_draws(pd.read_csv(io.BytesIO(uploaded_bytes)), validate=False)
        is_demo = False
    elif os.path.exists(path):
        df, balls_expanded = load_and_preprocess(path, validate=False)
        is_demo = False
    else:
        df, balls_expanded = load_sample_and_preprocess(n_draws=400)
        is_demo = True

    format_report = check_draw_format(df, balls_expanded)
    if current_format_only and format_report:
        keep = current_format_mask(df, balls_expanded)
        df = df[keep].reset_index(drop=True)
        balls_expanded = balls_expanded[keep].reset_index(drop=True)

    return df, balls_expanded, build_position_series(df, balls_expanded), is_demo, format_report


@st.cache_data(show_spinner=False)
def randomness_reports(position_series, n_columns):
    """All six per-position reports at once — recomputed only when the data changes."""
    return {p: randomness_report(position_series[p], p, n_columns) for p in range(n_columns)}


@st.cache_data(show_spinner=False)
def cached_gap_table(position_series, position, n_columns):
    return gap_table(position_series[position], position, n_columns)


@st.cache_data(show_spinner=False)
def cached_pooled_tests(balls_expanded, n_columns):
    return (
        pooled_uniformity_test(balls_expanded, main_positions(n_columns)),
        pooled_uniformity_test(balls_expanded, [super_position(n_columns)]),
        is_sorted_ascending(balls_expanded),
    )


@st.cache_data(show_spinner=False)
def cached_structure(balls_expanded):
    return structure_report(balls_expanded)


def essentials_card(n_draws, pooled_main, structure):
    """The answer, before the ten tabs that justify it.

    A dashboard that only answers questions you already know how to ask leaves a
    newcomer to assemble the conclusion out of ten tabs of p-values, and most
    people assemble the wrong one — they find the tab with the biggest number and
    stop. So the conclusion goes first, in words, and the tabs become the
    evidence for it rather than a puzzle to solve.

    Everything here is read off tests computed on the loaded data; none of it is
    a fixed string pretending to be a finding.
    """
    healthy = pooled_main["p_value"] > 0.05 and structure["looks_random"]
    mde = minimum_detectable_effect(n_draws)

    with st.container(border=True):
        st.markdown("### Lo esencial, antes de entrar en detalle")
        if healthy:
            st.markdown(
                f"**1. Tus datos están sanos.** Los {n_draws} sorteos cargados pasan las dos "
                "pruebas que importan: cada número sale con la frecuencia que le toca, y las "
                "combinaciones no se agrupan de ninguna forma rara. Es el resultado que quieres."
            )
        else:
            st.markdown(
                f"**1. Ojo con los datos.** Alguna de las pruebas sobre los {n_draws} sorteos "
                "cargados salió marcada. Lo más probable no es que la lotería sea vencible, sino "
                "que el archivo mezcle formatos o venga incompleto. Revisa la pestaña "
                "**5 · ¿Es aleatorio?** antes de creerle a nada más."
            )
        st.markdown(
            "**2. No hay nada que predecir, y eso no es un fallo del panel.** Un sorteo de Baloto "
            "es independiente y uniforme por diseño: las balotas no recuerdan lo que salió antes. "
            "Ningún modelo de esta página le gana al azar, y la pestaña **8 · ¿Le gana al azar?** "
            "lo mide sobre tus propios datos en lugar de pedirte que lo creas."
        )
        st.markdown(
            f"**3. Con {n_draws} sorteos, «no encontré nada» quiere decir "
            f"«no hay ventaja mayor que +{mde['relative']:.0%}».** No quiere decir «no hay "
            "ventaja». Esa distinción es la pestaña **9 · ¿Qué se podía ver?**, y es lo que "
            "separa este panel de un sistema de lotería."
        )
        st.markdown(
            "**4. Lo único que sí puedes mejorar es cuánto cobras si ganas.** Tu probabilidad de "
            "ganar es fija e idéntica para toda combinación. Pero el premio mayor se reparte entre "
            "todos los que acertaron, y la gente no elige al azar — juega fechas. Evitar lo que "
            "juegan los demás no te hace ganar más seguido; te hace repartir con menos gente. "
            "Está en **7 · Jugadas → Reparto de premios**."
        )


def _bucketed_sums(table, width=10):
    """The exact sum table folded into readable buckets.

    191 attainable sums over a few hundred draws puts ~2 draws in each, and a
    191-bar chart of twos and threes shows a reader nothing but noise. Bucketing
    the *exact* expected counts alongside the observed ones keeps the comparison
    honest — the reference is still combinatorial, just summed over a wider bin.
    """
    edges = list(range(int(table["sum"].min()), int(table["sum"].max()) + width + 1, width))
    buckets = pd.cut(table["sum"], bins=edges, right=False)
    grouped = table.groupby(buckets, observed=True).agg(
        observed=("observed", "sum"), expected=("expected", "sum"),
        low=("sum", "min"), high=("sum", "max"))
    grouped = grouped.reset_index(drop=True)
    grouped["label"] = grouped["low"].astype(int).astype(str) + "-" + grouped["high"].astype(int).astype(str)
    return grouped


def render_structure(structure):
    """The three order-agnostic summaries against their exact distributions.

    Lives inside the randomness tab rather than in one of its own because it
    answers the same question — is there anything here to exploit? — from the
    one angle the pooled test structurally cannot reach: dependence *between*
    the five balls rather than the frequency of each.
    """
    section("¿Se combinan bien las balotas, o solo se reparten bien?", "structure_section")

    tests = structure["tests"]
    c1, c2, c3 = st.columns(3)
    for column, key, label in (
        (c1, "sum", "Suma de las 5"),
        (c2, "parity", "Cuántas impares"),
        (c3, "calendar", f"Cuántas ≤ {CALENDAR_THRESHOLD}"),
    ):
        p_value = tests[key]["p_value"]
        column.metric(label, "Normal" if p_value > 0.05 else "Marcada",
                      help=HELP["structure_verdict"])
        # Not `delta=`: Streamlit draws an up arrow next to it, and a p-value has
        # no direction to point in.
        column.caption(f"p = {p_value:.3f}")

    plain_verdict(
        structure["looks_random"],
        "Las combinaciones salen como la combinatoria manda"
        if structure["looks_random"] else
        "Alguna de las tres pruebas salió marcada",
        f"Suma media observada {structure['observed_mean_sum']:.1f} contra "
        f"{structure['expected_mean_sum']:.1f} exacta, sobre {structure['n_draws']} sorteos. "
        "Con tres pruebas a la vez, que una marque «No» le pasa a un histórico impecable "
        "alrededor del 14% de las veces — una sola casilla no es un hallazgo.",
    )

    bucketed = _bucketed_sums(structure["sum_distribution"])
    figure = go.Figure()
    figure.add_bar(x=bucketed["label"], y=bucketed["observed"], name="Tus sorteos")
    figure.add_scatter(x=bucketed["label"], y=bucketed["expected"], mode="lines+markers",
                       name="Exacto (combinatoria)")
    figure.update_layout(xaxis_title="Suma de las 5 balotas", yaxis_title="Sorteos", height=380)
    chart(figure, "Suma de las balotas: tus sorteos contra lo exacto", "sum_chart")
    st.caption(
        f"Hay **una sola** combinación que suma {int(structure['sum_distribution']['sum'].min())} "
        f"(1-2-3-4-5) y **{int(structure['sum_distribution']['n_combinations'].max()):,}** que suman "
        f"{int(structure['expected_mean_sum'])}. Por eso la campana. Y sin embargo cada una de esas "
        "combinaciones es exactamente igual de probable que 1-2-3-4-5: la campana cuenta cuántas "
        "hay, no cuánto valen. Jugar una suma rara no mejora tu probabilidad de ganar — solo "
        "reduce con cuánta gente repartirías."
    )

    c1, c2 = st.columns(2)
    with c1:
        parity = structure["parity_distribution"]
        figure = go.Figure()
        figure.add_bar(x=parity["n_odd"], y=parity["observed"], name="Tus sorteos")
        figure.add_scatter(x=parity["n_odd"], y=parity["expected"], mode="markers",
                           name="Exacto", marker=dict(size=11, symbol="diamond"))
        figure.update_layout(xaxis_title="Balotas impares (de 5)", yaxis_title="Sorteos", height=320)
        chart(figure, "Pares e impares", "parity_chart")
    with c2:
        calendar = structure["calendar_distribution"]
        figure = go.Figure()
        figure.add_bar(x=calendar["n_at_or_below"], y=calendar["observed"], name="Tus sorteos")
        figure.add_scatter(x=calendar["n_at_or_below"], y=calendar["expected"], mode="markers",
                           name="Exacto", marker=dict(size=11, symbol="diamond"))
        figure.update_layout(xaxis_title=f"Balotas de {CALENDAR_THRESHOLD} o menos (de 5)",
                             yaxis_title="Sorteos", height=320)
        chart(figure, "Números que caben en una fecha", "calendar_chart")

    st.caption(
        f"El reparto de la derecha es el que tiene consecuencias económicas. Los números "
        f"1-{CALENDAR_THRESHOLD} son el {calendar.attrs['pool_share']:.0%} del pool pero se juegan "
        "muchísimo más que eso, porque caben en un cumpleaños. Un sorteo con 4 o 5 números por "
        "encima de 31 paga igual de seguido y se reparte entre menos gente — ver "
        "**7 · Jugadas → Reparto de premios**."
    )


def render():
    st.title("🎯 Baloto")
    st.caption(
        "Panel de análisis y forecasting para Baloto. Las pestañas están numeradas en el orden en que "
        "conviene leerlas: los sorteos de lotería son, por diseño, independientes y uniformes, así que "
        "el objetivo de este panel es mostrar honestamente si hay o no señal explotable — no prometer "
        "que la hay. Si una palabra no te suena, está en el **Glosario** de la barra lateral."
    )

    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader("CSV propio (columnas Date, Ball)", type="csv")
        data_path = st.text_input("Ruta local (si no subes archivo)", value=DEFAULT_DATA_PATH)
        current_format_only = st.checkbox(
            "Solo sorteos del formato actual", value=True,
            help="Baloto cambió de reglas en abril de 2017 (antes: 6 balotas del 1 al 45, sin "
                 "superbalota). Los dos formatos se publican igual, así que un histórico largo suele "
                 "mezclarlos. Desmarca solo si sabes lo que estás haciendo.",
        )
    glossary()

    df, balls_expanded, position_series, is_demo, format_report = load_data(
        data_path, uploaded.getvalue() if uploaded else None, current_format_only
    )
    n_columns = balls_expanded.shape[1]
    n_draws = len(df)
    label_to_pos = {series_label(p, n_columns): p for p in range(n_columns)}

    if format_report:
        if current_format_only:
            st.info(
                f"Se descartaron **{format_report['n_dropped_by_cutoff']} de {format_report['n_draws']} "
                f"sorteos** anteriores al cambio de reglas de 2017 (el juego antiguo sacaba 6 balotas del "
                f"1 al 45, sin superbalota). Se analizan los **{format_report['n_current_format']} sorteos "
                f"del formato actual**, desde {format_report['current_era_starts']:%Y-%m-%d}. "
                "Puedes desactivar el filtro en la barra lateral."
            )
        else:
            st.error(
                f"**Tus datos mezclan dos juegos distintos.** {format_report['n_violations']} de "
                f"{format_report['n_draws']} sorteos (entre {format_report['first_violation']:%Y-%m-%d} y "
                f"{format_report['last_violation']:%Y-%m-%d}) contienen números que el juego actual no "
                f"puede producir. Todo lo que sigue — frecuencias, hot/cold, pruebas de aleatoriedad, "
                "backtest — está calculado sobre esa mezcla y no es interpretable. Marca **Solo sorteos "
                "del formato actual** en la barra lateral."
            )

    if is_demo:
        st.info(
            "No se encontró un CSV real en `exported_data/` ni se subió uno propio — mostrando **datos sintéticos "
            "de demostración** (sorteos uniformes independientes generados aleatoriamente), solo para que puedas "
            "explorar el panel. Sube tu CSV real en la barra lateral para analizar tus datos."
        )

    pooled_main, pooled_super, sorted_flag = cached_pooled_tests(balls_expanded, n_columns)
    structure = cached_structure(balls_expanded)
    essentials_card(n_draws, pooled_main, structure)

    # The numbers are a reading order, not decoration. Ten tabs in a row with no
    # ordering reads as ten equally good places to start, and the one a newcomer
    # picks first is usually "Forecast" — the single tab whose output means least
    # without the three that come before it.
    tabs = st.tabs([
        "1 · Resumen", "2 · Probabilidades", "3 · Frecuencia", "4 · Hot / Cold",
        "5 · ¿Es aleatorio?", "6 · Pronóstico", "7 · Jugadas", "8 · ¿Le gana al azar?",
        "9 · ¿Qué se podía ver?", "10 · Registro",
    ])

    # ---------------------------------------------------------------- Resumen
    with tabs[0]:
        section("Estado de los datos", "tab_resumen")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sorteos", n_draws, help=HELP["n_draws"])
        col2.metric("Desde", df["ds"].min().strftime("%Y-%m-%d"), help=HELP["date_from"])
        col3.metric("Hasta", df["ds"].max().strftime("%Y-%m-%d"), help=HELP["date_to"])
        col4.metric("Balotas guardadas ordenadas asc.", "Sí" if sorted_flag else "No",
                    help=HELP["sorted_flag"])

        if sorted_flag:
            st.warning(
                "Tus datos parecen tener las 5 balotas principales guardadas de menor a mayor por sorteo. Eso "
                "convierte cada columna en un **estadístico de orden** (mínimo, 2do menor, ...), no en una balota "
                "uniforme — un chi-cuadrado por posición puede marcar 'no aleatorio' solo por el orden, no porque "
                "haya un patrón real. Usa la prueba agrupada (pooled) de abajo, que es inmune a esto."
            )

        section(
            f"¿Los números principales ({MAIN_BALL_RANGE[0]}-{MAIN_BALL_RANGE[1]}) se reparten uniformemente?",
            "pooled_test",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Las 5 balotas principales (1-43)**", help=HELP["pooled_main_p"])
            plain_verdict(
                pooled_main["p_value"] > 0.05,
                "Cada número sale las veces que le toca"
                if pooled_main["p_value"] > 0.05 else
                "Hay un desbalance que el azar no explica",
                f"p-valor = {pooled_main['p_value']:.3f}. Por encima de 0.05 significa «nada raro»; "
                "por debajo, «esto sería difícil de conseguir con un sorteo justo» — y casi siempre "
                "apunta a un problema del archivo antes que a una lotería vencible.",
            )
        with c2:
            st.markdown("**La superbalota (1-16)**", help=HELP["pooled_super_p"])
            plain_verdict(
                pooled_super["p_value"] > 0.05,
                "Sale repartida como debe"
                if pooled_super["p_value"] > 0.05 else
                "Hay un desbalance que el azar no explica",
                f"p-valor = {pooled_super['p_value']:.3f}. Se evalúa aparte porque su rango es 1-16 "
                "y no 1-43, así que no se puede agrupar con las principales.",
            )

    # ------------------------------------------- Probabilidades y Valor Esperado
    with tabs[1]:
        section("Combinatoria exacta", "tab_probabilidades")
        st.markdown(
            "Aquí no hay nada que predecir: las probabilidades de cada categoría son **combinatoria exacta**. "
            f"Un tiquete son {MAIN_BALLS_DRAWN} números de {MAIN_BALL_RANGE[0]}-{MAIN_BALL_RANGE[1]} más una "
            f"superbalota de {SUPER_BALL_RANGE[0]}-{SUPER_BALL_RANGE[1]}, así que hay "
            f"**{total_combinations():,}** tiquetes igualmente probables. Lo único que hace falta para saber "
            "cuánto vale jugar es la tabla de premios vigente."
        )

        prob_table = category_probabilities()
        jackpot_odds = prob_table.loc[
            (prob_table["main_matches"] == 5) & (prob_table["super_match"]), "odds_one_in"
        ].iloc[0]

        c1, c2 = st.columns(2)
        c1.metric("Probabilidad del premio mayor", f"1 en {jackpot_odds:,.0f}", help=HELP["jackpot_odds"])
        c2.metric("Combinaciones posibles", f"{total_combinations():,}", help=HELP["total_combos"])

        section("Tabla de premios", "prize_table")
        st.caption(
            "Los montos de abajo son **valores de ejemplo que debes reemplazar** con la tabla oficial vigente "
            "(varias categorías son variables y el premio mayor se acumula). Pon 0 en las categorías que no "
            "pagan premio. Las probabilidades sí son exactas y no dependen de lo que escribas aquí."
        )

        ticket_price = st.number_input("Precio del tiquete (COP)", min_value=0, value=5700, step=100,
                                        help=HELP["ticket_price"])

        default_payouts = {
            (5, True): 5_000_000_000, (5, False): 80_000_000,
            (4, True): 8_000_000, (4, False): 400_000,
            (3, True): 100_000, (3, False): 20_000,
            (2, True): 10_000, (2, False): 0,
            (1, True): 5_700, (1, False): 0,
            (0, True): 5_700, (0, False): 0,
        }
        editor_df = prob_table[["category", "main_matches", "super_match", "probability", "odds_one_in"]].copy()
        editor_df["payout"] = [
            default_payouts.get((int(r.main_matches), bool(r.super_match)), 0) for r in editor_df.itertuples()
        ]

        edited = st.data_editor(
            editor_df,
            column_config={
                "category": st.column_config.TextColumn("Categoría", disabled=True),
                "main_matches": None,
                "super_match": None,
                "probability": st.column_config.NumberColumn("Probabilidad", format="%.8f", disabled=True),
                "odds_one_in": st.column_config.NumberColumn("1 en...", format="%.0f", disabled=True),
                "payout": st.column_config.NumberColumn("Premio (COP)", min_value=0, step=1000),
            },
            hide_index=True,
            use_container_width=True,
        )

        payouts = {
            (int(r.main_matches), bool(r.super_match)): float(r.payout) for r in edited.itertuples()
        }
        ev = expected_value(prob_table, payouts, ticket_price)

        section("Valor esperado por tiquete", "ev_section")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retorno esperado", f"${ev['expected_return']:,.0f}", help=HELP["expected_return"])
        m2.metric("Valor esperado", f"${ev['expected_value']:,.0f}",
                  delta=f"{ev['expected_value']:,.0f} por tiquete", help=HELP["expected_value"])
        m3.metric("Retorno al jugador (RTP)", f"{ev['return_to_player'] * 100:.1f}%", help=HELP["rtp"])
        m4.metric("Prob. de ganar algo", f"1 en {ev['odds_any_prize_one_in']:,.1f}", help=HELP["any_prize"])

        if ev["expected_value"] < 0:
            st.error(
                f"Con esta tabla de premios, cada tiquete pierde en promedio **${abs(ev['expected_value']):,.0f}**. "
                "Esto no es opinión ni un modelo: es el valor esperado exacto. Ninguna estrategia de selección de "
                "números lo cambia, porque todas las combinaciones son igual de probables."
            )
        else:
            st.warning(
                "El valor esperado sale positivo con estos montos — revisa que los premios sean los reales. "
                "Aun cuando un acumulado grande lo vuelve positivo en el papel, el premio mayor se reparte entre "
                "todos los ganadores (y hay retención en la fuente), así que el retorno real suele ser menor."
            )

        breakeven = breakeven_jackpot(prob_table, payouts, ticket_price)
        st.metric("Premio mayor necesario para que el valor esperado sea cero", f"${breakeven:,.0f}",
                  help=HELP["breakeven"])

        plot_df = prob_table[prob_table["probability"] > 0].copy()
        fig = go.Figure()
        fig.add_bar(x=plot_df["category"], y=plot_df["probability"])
        fig.update_layout(yaxis_type="log", yaxis_title="Probabilidad (escala log)", xaxis_title="Categoría")
        chart(fig, "Probabilidad exacta de cada categoría de premio", "category_chart")

    # ---------------------------------------------------------- Frecuencia y Gaps
    with tabs[2]:
        section("Frecuencia y atrasos", "tab_frecuencia")
        chosen_label = st.selectbox("Posición", list(label_to_pos.keys()), key="freq_pos")
        pos = label_to_pos[chosen_label]

        freq = frequency_table(position_series[pos], pos, n_columns)
        fig = go.Figure()
        fig.add_bar(x=freq["number"], y=freq["count"], name="Observado")
        fig.add_hline(y=float(freq["expected_count"].iloc[0]), line_dash="dash",
                      annotation_text="Esperado (uniforme)", line_color="gray")
        fig.update_layout(xaxis_title="Número", yaxis_title="Veces salido")
        chart(fig, f"Frecuencia — {chosen_label}", "freq_chart")

        section("Gaps entre apariciones y 'número atrasado'", "gaps_section")
        st.caption(
            "El *overdue score* es el heurístico popular de 'este número ya se demoró, debe salir'. Para sorteos "
            "independientes no tiene poder predictivo real — es la falacia del jugador — pero se incluye porque es "
            "una vista que mucha gente busca. Interprétalo como curiosidad, no como señal."
        )
        gaps = cached_gap_table(position_series, pos, n_columns)
        st.dataframe(
            gaps.sort_values("overdue_score", ascending=False)
            .style.format({"avg_gap_days": "{:.1f}", "std_gap_days": "{:.1f}", "overdue_score": "{:.2f}"}),
            use_container_width=True,
        )

    # ---------------------------------------------------------------- Hot/Cold
    with tabs[3]:
        section("Números calientes y fríos", "tab_hotcold")
        chosen_label_hc = st.selectbox("Posición", list(label_to_pos.keys()), key="hc_pos")
        pos_hc = label_to_pos[chosen_label_hc]
        window = st.slider("Ventana reciente (# sorteos)", 5, 60, 20, help=HELP["hotcold_window"])

        hc = hot_cold_numbers(position_series[pos_hc], pos_hc, n_columns, recent_draws=window)
        fig = go.Figure()
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in hc["delta_pct"]]
        fig.add_bar(x=hc["number"], y=hc["delta_pct"], marker_color=colors)
        fig.update_layout(xaxis_title="Número", yaxis_title="Diferencia de participación (%)")
        chart(fig, f"Hot (rojo) / Cold (azul) — {chosen_label_hc} (últimos {window} sorteos vs. histórico)",
              "hotcold_chart")
        st.caption(
            "Diferencia entre el % de apariciones en la ventana reciente y el % histórico. Con pocas observaciones "
            "por ventana, este ruido es esperable incluso sin ningún cambio real en el proceso de sorteo."
        )

    # ------------------------------------------------------------ Aleatoriedad
    with tabs[4]:
        section("¿Hay alguna estructura explotable?", "tab_aleatoriedad")
        section("Veredicto por posición", "verdict_table")
        reports = randomness_reports(position_series, n_columns)
        rows = []
        for rep in reports.values():
            rows.append({
                "Posición": rep["label"],
                "Sorteos": rep["n_draws"],
                "chi2 p-valor": rep["chi_square"]["p_value"],
                "runs test p-valor": rep["runs_test"]["p_value"],
                "Ljung-Box p-valor": rep["autocorrelation"]["ljung_box_p_value"],
                "Parece aleatorio": "Sí" if rep["looks_random"] else "No",
            })
        report_df = pd.DataFrame(rows)
        st.dataframe(report_df.style.format({
            "chi2 p-valor": "{:.3f}", "runs test p-valor": "{:.3f}", "Ljung-Box p-valor": "{:.3f}",
        }), use_container_width=True)
        st.caption(
            "Con 6 posiciones evaluadas a la vez, es normal que 1 de cada ~20 pruebas marque 'No' solo por azar "
            "(comparaciones múltiples) — no lo tomes como señal a menos que se repita de forma consistente y esté "
            "confirmado por la prueba agrupada (pestaña Resumen)."
        )

        section("Autocorrelación (ACF)", "acf_section")
        acf_label = st.selectbox("Posición", list(label_to_pos.keys()), key="acf_pos")
        acf_pos = label_to_pos[acf_label]
        autocorr = reports[acf_pos]["autocorrelation"]  # already computed above
        n = len(position_series[acf_pos])
        band = 1.96 / (n ** 0.5)
        fig = go.Figure()
        fig.add_bar(x=list(range(len(autocorr["acf"]))), y=autocorr["acf"], name="ACF")
        fig.add_hline(y=band, line_dash="dash", line_color="gray")
        fig.add_hline(y=-band, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="Lag", yaxis_title="Autocorrelación")
        chart(fig, f"ACF — {acf_label}", "acf_chart")
        st.metric("Ljung-Box p-valor (¿hay autocorrelación?)", f"{autocorr['ljung_box_p_value']:.3f}",
                  help=HELP["ljung_box"])
        verdict_badge(autocorr["ljung_box_p_value"] > 0.05,
                      "Sin autocorrelación detectable — no hay 'memoria' que un modelo de series de tiempo pueda explotar",
                      "Autocorrelación detectada — esto sí justificaría probar un modelo de series de tiempo")

        st.divider()
        render_structure(structure)

    # ------------------------------------------------------------------ Forecast
    with tabs[5]:
        section("Sugerencia para el próximo sorteo", "tab_forecast")
        st.warning(
            "Estos son ejercicios de forecasting, no predicciones confiables: para un sorteo justo, ningún modelo "
            "puede superar de forma sostenida la probabilidad teórica. Revisa la pestaña Backtest antes de confiar "
            "en cualquiera de estos números."
        )
        model_options = ["FrequencyBaseline", "Prophet", *MODEL_NAMES, "XGBoost", "TimesFM"]
        if not PROPHET_AVAILABLE:
            model_options.remove("Prophet")
        if not TIMESFM_AVAILABLE:
            model_options.remove("TimesFM")
        model_choice = st.selectbox("Modelo", model_options, help=HELP["model_choice"])
        if not PROPHET_AVAILABLE:
            st.caption(HELP["prophet_missing"])
        if not TIMESFM_AVAILABLE:
            st.caption(HELP["timesfm_missing"])
        elif model_choice == "TimesFM":
            st.caption(HELP["timesfm_first_run"])

        if st.button("Generar predicción del próximo sorteo"):
            with st.spinner("Entrenando..."):
                history = position_series[0]["ds"]
                next_date = next_draw_dates(history.max(), 1, weekdays=infer_draw_weekdays(history))[0]

                preds = {}
                if model_choice == "FrequencyBaseline":
                    preds = most_frequent_pick(position_series)
                elif model_choice == "Prophet":
                    from lottery.models.prophet_model import define_and_fit_model, predict_at_dates
                    for p in range(n_columns):
                        m = define_and_fit_model(position_series[p])  # sin festivos: no afectan una balota
                        # predict_at_dates evalúa solo la fecha pedida; make_predictions
                        # re-predeciría toda la historia para usar una sola fila.
                        fc = predict_at_dates(m, p, n_columns, [next_date])
                        preds[p] = int(fc["yhat_adjusted"].iloc[0])
                elif model_choice in MODEL_NAMES:
                    raw = fit_predict_all(position_series, h=1)
                    clipped = adjusted_predictions(raw, n_columns, model_name=model_choice)
                    preds = dict(zip(clipped["unique_id"].astype(int), clipped["yhat_adjusted"].astype(int), strict=True))
                elif model_choice == "XGBoost":
                    for p in range(n_columns):
                        preds[p] = forecast_next(position_series[p], p, n_columns, next_date)
                elif model_choice == "TimesFM":
                    # Zero-shot: no fit, one forward pass over every position at
                    # once. `next_date` is not passed because TimesFM reads values,
                    # not dates — it sits on the sequential draw axis, like the
                    # statsforecast models and unlike Prophet.
                    try:
                        preds = timesfm_forecast(position_series, n_columns) or {}
                    except CheckpointUnavailableError as exc:
                        # The expected failure, not a bug: the weights download on
                        # first use. Shown as a sentence with the original error as
                        # detail, the same shape the football and cycling pages use
                        # for their domain refusals.
                        st.error(HELP["timesfm_download_failed"])
                        st.caption(str(exc))
                        preds = None

            # preds is None only when the block above already put an explanation
            # on screen; anything else here would print a second, vaguer one.
            if preds is None:
                st.caption("")
            elif any(preds.get(p) is None for p in range(n_columns)):
                st.error("No hay suficiente historia para entrenar este modelo. Carga un CSV con más sorteos.")
            else:
                main_numbers = sorted({preds[p] for p in main_positions(n_columns)})
                super_number = preds[n_columns - 1]
                st.success(f"Balotas principales sugeridas: **{' - '.join(str(n) for n in main_numbers)}**  |  "
                           f"Superbalota: **{super_number}**")
                if len(main_numbers) < MAIN_BALLS_DRAWN:
                    st.caption(f"Nota: hubo {MAIN_BALLS_DRAWN - len(main_numbers)} coincidencia(s) entre posiciones, "
                               "por eso hay menos de 5 números distintos — típico cuando el modelo no tiene señal real "
                               "que diferencie una posición de otra.")

    # ---------------------------------------------------------------- Jugadas
    with tabs[6]:
        section("Generar, verificar y medir jugadas", "tab_jugadas")
        st.markdown(
            "Genera jugadas, verifícalas contra los sorteos reales y mide si tu forma de generarlas le gana "
            "al azar. **Generar números es perfectamente válido** — lo que ninguna estrategia puede hacer es "
            "producir una jugada *más probable* que otra, porque las "
            f"{total_combinations():,} combinaciones son igual de probables. Esa afirmación no te pedimos que "
            "la creas: la pestaña la mide sobre tus propios datos."
        )

        gen_tab, check_tab, exp_tab, split_tab = st.tabs(
            ["Generar", "Verificar", "Medir estrategias", "Reparto de premios"])

        # ------------------------------------------------------------- generar
        with gen_tab:
            c1, c2, c3 = st.columns(3)
            n_tickets = c1.slider("Cuántas jugadas", 1, 20, 5, help=HELP["n_tickets"])
            strategy = c2.selectbox("Estrategia", list(STRATEGIES),
                                    help="'random' es la honesta: todas las combinaciones son igual de probables.")
            disjoint = c3.checkbox("Sin números repetidos entre jugadas", value=True,
                                   help="Reparte las jugadas sobre más números del pool.")

            if st.button("Generar jugadas"):
                tickets = generate_portfolio(n_tickets, strategy=strategy,
                                              balls_expanded=balls_expanded, disjoint=disjoint)
                st.session_state["tickets"] = tickets

            if "tickets" in st.session_state:
                tickets = st.session_state["tickets"]
                st.dataframe(pd.DataFrame([
                    {"#": i + 1, "Balotas": " - ".join(str(n) for n in sorted(t.main)),
                     "Superbalota": t.super_ball, "Popularidad": popularity_score(t)}
                    for i, t in enumerate(tickets)
                ]).style.format({"Popularidad": "{:.3f}"}),
                    use_container_width=True, hide_index=True)
                st.caption(
                    "La columna **Popularidad** no dice nada sobre tus probabilidades de ganar, que son "
                    "idénticas para toda combinación. Dice qué tan seguido juega otra gente esos "
                    "números, y por lo tanto entre cuántos repartirías. Ver la pestaña *Reparto de "
                    "premios*.",
                    help=HELP["popularity_score"],
                )

                cov = portfolio_coverage(tickets)
                m1, m2, m3 = st.columns(3)
                m1.metric("Números distintos cubiertos", f"{cov['distinct_main_numbers']} de {MAIN_BALL_RANGE[1]}",
                          help=HELP["distinct_numbers"])
                m2.metric("Cobertura del pool", f"{cov['pool_coverage_pct']:.0f}%", help=HELP["pool_coverage"])
                m3.metric("Prob. de premio mayor", f"1 en {cov['jackpot_odds_one_in']:,.0f}",
                          help=HELP["portfolio_odds"])
                st.caption(
                    "Lo único que cambia al jugar varias combinaciones distintas es **cuántas** posibilidades "
                    "compras, no la calidad de ninguna. Comprar N jugadas divide las probabilidades del premio "
                    "mayor entre N — eso es aritmética, no una estrategia."
                )

        # ----------------------------------------------------------- verificar
        with check_tab:
            st.caption("Escribe una jugada y mira cómo le habría ido en todos los sorteos de tu histórico.",
                       help=HELP["check_input"])
            c1, c2 = st.columns([3, 1])
            main_text = c1.text_input(
                f"5 balotas ({MAIN_BALL_RANGE[0]}-{MAIN_BALL_RANGE[1]}), separadas por coma o guion", "3, 12, 19, 27, 41")
            super_text = c2.number_input("Superbalota", min_value=SUPER_BALL_RANGE[0],
                                          max_value=SUPER_BALL_RANGE[1], value=8)

            if st.button("Verificar jugada"):
                try:
                    numbers = tuple(int(x) for x in main_text.replace("-", ",").split(",") if x.strip())
                    ticket = Ticket(main=numbers, super_ball=int(super_text))
                except (ValueError, TypeError) as exc:
                    st.error(f"Jugada inválida: {exc}")
                else:
                    last_main, last_super = draw_from_row(balls_expanded.iloc[-1])
                    last = check_ticket(ticket, last_main, last_super)
                    st.metric(f"Último sorteo ({df['ds'].max():%Y-%m-%d})", last["category"])
                    if last["matched_numbers"]:
                        st.write("Números acertados:", ", ".join(str(n) for n in last["matched_numbers"]))

                    results = check_against_history(ticket, df, balls_expanded)
                    st.subheader(f"Historial completo — {len(results)} sorteos")
                    summary = history_summary(results)
                    fig = go.Figure()
                    fig.add_bar(x=summary["category"], y=summary["times"])
                    fig.update_layout(xaxis_title="Categoría", yaxis_title="Sorteos")
                    chart(fig, "En cuántos sorteos habría caído cada categoría", "history_chart")
                    st.dataframe(summary.style.format({"share_pct": "{:.2f}"}),
                                 use_container_width=True, hide_index=True)
                    st.caption(
                        f"Mejor resultado histórico de esta jugada: **{int(results['main_matches'].max())} aciertos**. "
                        "Cualquier otra jugada habría dado una distribución estadísticamente equivalente."
                    )

        # ------------------------------------------------------ medir estrategias
        with exp_tab:
            section("El experimento", "experiment_intro")
            st.markdown(
                "El experimento: para cada sorteo histórico se generan jugadas usando **solo** los sorteos "
                "anteriores, y se comparan los aciertos contra la expectativa exacta del azar "
                "(hipergeométrica). Si una estrategia tuviera ventaja real, el p-valor sería pequeño."
            )
            c1, c2 = st.columns(2)
            max_back = max(10, n_draws - MIN_TRAIN_FLOOR - 1)
            draws_back = c1.slider("Sorteos a evaluar", 10, max_back, min(200, max_back),
                                   help=HELP["draws_back"])
            per_draw = c2.slider("Jugadas por sorteo", 1, 50, 10, help=HELP["per_draw"])


            if st.button("Ejecutar experimento"):
                with st.spinner("Generando y puntuando jugadas..."):
                    try:
                        st.session_state["strategy_table"] = compare_strategies(
                            df, balls_expanded, n_draws_back=draws_back,
                            tickets_per_draw=per_draw, min_history=MIN_TRAIN_FLOOR, seed=42)
                    except ValueError as exc:
                        st.error(str(exc))

            if "strategy_table" in st.session_state:
                table = st.session_state["strategy_table"]
                fig = go.Figure()
                fig.add_bar(x=table["strategy"], y=table["avg_main_matches"], name="Estrategia")
                fig.add_bar(x=table["strategy"], y=table["chance_avg_main_matches"], name="Azar (esperado)")
                fig.update_layout(barmode="group", yaxis_title="Aciertos promedio")
                chart(fig, "Aciertos promedio por jugada vs. azar", "strategy_chart")

                display = table.copy()
                display["¿Le gana al azar?"] = display["beats_chance_corrected"].map({True: "Sí", False: "No"})
                display["Ventaja (IC 95%)"] = display.apply(
                    lambda r: f"{r['effect']:+.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1)
                st.markdown("**Resultado por estrategia**", help=HELP["strategy_table"])
                st.dataframe(
                    display[["strategy", "n_tickets_evaluated", "avg_main_matches",
                             "chance_avg_main_matches", "Ventaja (IC 95%)",
                             "p_value_better_than_chance", "¿Le gana al azar?", "best_result"]]
                    .style.format({"avg_main_matches": "{:.4f}", "chance_avg_main_matches": "{:.4f}",
                                    "p_value_better_than_chance": "{:.4f}"}),
                    use_container_width=True, hide_index=True)
                st.caption(
                    f"La columna de veredicto usa un umbral corregido por comparaciones múltiples "
                    f"(Bonferroni, {table['bonferroni_threshold'].iloc[0]:.4f}): al probar varias estrategias a la "
                    "vez, alguna parecerá ganadora por puro azar con más frecuencia de lo que sugiere un 0.05 "
                    "suelto."
                )

            st.divider()
            section("¿El resultado se sostiene?", "stability_section")
            st.markdown(
                "Una sola corrida es **un** sorteo de un proceso ruidoso: con α = 0.05, una estrategia sin "
                "ninguna ventaja igual parece ganadora ~1 de cada 20 veces. Así es como la mayoría de la gente "
                "se convence de que su sistema funciona. Esto repite el experimento con varias semillas y cuenta "
                "cuántas veces marcó ganador."
            )
            seeds = st.slider("Semillas a probar", 5, 50, 20,
                              help="Cada semilla es una repetición independiente del experimento.")
            if st.button("Comprobar estabilidad"):
                with st.spinner(f"Repitiendo el experimento {seeds} veces por estrategia..."):
                    try:
                        st.session_state["stability"] = pd.DataFrame([
                            stability_check(df, balls_expanded, strategy=s, n_seeds=seeds,
                                            n_draws_back=draws_back, tickets_per_draw=per_draw,
                                            min_history=MIN_TRAIN_FLOOR)
                            for s in STRATEGIES
                        ])
                    except ValueError as exc:
                        st.error(str(exc))

            if "stability" in st.session_state:
                stab = st.session_state["stability"].copy()
                stab["Marcada ganadora"] = stab.apply(
                    lambda r: f"{r['times_flagged']} de {r['n_seeds']} ({r['flag_rate']:.0%})", axis=1)
                st.markdown("**Tasa de falsos positivos medida**", help=HELP["stability_table"])
                st.dataframe(
                    stab[["strategy", "Marcada ganadora", "expected_flag_rate_if_no_edge", "median_p_value"]]
                    .rename(columns={"strategy": "Estrategia",
                                      "expected_flag_rate_if_no_edge": "Esperado sin ventaja",
                                      "median_p_value": "p mediano"})
                    .style.format({"Esperado sin ventaja": "{:.0%}", "p mediano": "{:.3f}"}),
                    use_container_width=True, hide_index=True)
                st.caption(
                    "`random` no puede tener ventaja: es la referencia. Si otra estrategia no marca ganador "
                    "claramente más seguido que ella, no ha demostrado nada."
                )

        # ------------------------------------------------------- reparto de premios
        with split_tab:
            section("El premio mayor se reparte", "tab_split")
            st.markdown(
                "Todo lo demás en este panel termina igual: nada cambia tu **probabilidad** de ganar, "
                f"porque las {total_combinations():,} combinaciones son igual de probables. Eso sigue siendo cierto "
                "aquí. Lo que cambia es la otra mitad del valor esperado.\n\n"
                "El premio mayor se reparte entre todos los que tengan la combinación ganadora, y la "
                "gente **no elige al azar**: juega cumpleaños (por eso el 1-31 va sobrecargado y el "
                "32-43 abandonado), rachas seguidas, patrones sobre la grilla del tiquete y números de "
                "la suerte. Una combinación popular y una impopular ganan exactamente igual de seguido; "
                "condicionado a ganar, la popular paga una fracción de la otra.\n\n"
                "`P(ganar)` → no la mueve nada, nunca.  \n"
                "`E[premio | ganar]` → esto sí se puede mejorar, evitando lo que juegan los demás."
            )

            st.warning(
                "**Qué es este modelo, con precisión.** Una función heurística sobre los sesgos que "
                "aparecen en la literatura de elección de números, con pesos que este proyecto **no "
                "puede calibrar**: haría falta saber qué tiquetes compró la gente, y ningún operador lo "
                "publica. La *dirección* es sólida (las fechas se sobrejuegan, y el efecto es grande); "
                "cualquier número concreto es una estimación gruesa. Por eso el resultado viene con "
                "banda y no con una sola cifra."
            )

            c1, c2, c3 = st.columns(3)
            jackpot = c1.number_input("Premio mayor acumulado (COP)", min_value=0,
                                       value=5_000_000_000, step=100_000_000)
            tickets_sold = c2.number_input("Tiquetes vendidos para el sorteo", min_value=1,
                                            value=3_000_000, step=100_000, help=HELP["tickets_sold"])
            multiplier = c3.slider("Cuánto más se juega la combinación más popular", 2.0, 25.0, 10.0,
                                    step=1.0, help=HELP["multiplier"])

            default_tickets = st.session_state.get("tickets")
            examples = [
                Ticket(main=(3, 7, 12, 19, 25), super_ball=8),
                Ticket(main=(1, 2, 3, 4, 5), super_ball=6),
                Ticket(main=(5, 10, 15, 20, 25), super_ball=7),
                Ticket(main=(33, 36, 38, 41, 43), super_ball=14),
            ]
            source = st.radio("Qué comparar", ["Ejemplos ilustrativos", "Mis jugadas generadas"],
                               horizontal=True, key="split_source")
            if source == "Mis jugadas generadas" and not default_tickets:
                st.info("Todavía no has generado jugadas — hazlo en la pestaña *Generar*.")
                to_compare = examples
            else:
                to_compare = examples if source == "Ejemplos ilustrativos" else default_tickets

            table = compare_tickets(to_compare, jackpot, tickets_sold,
                                     popularity_multiplier=multiplier)
            fig = go.Figure()
            fig.add_bar(x=table["ticket"], y=table["expected_jackpot_share"], name="Premio esperado",
                        error_y=dict(type="data", symmetric=False,
                                     array=table["share_high"] - table["expected_jackpot_share"],
                                     arrayminus=table["expected_jackpot_share"] - table["share_low"]))
            fig.add_hline(y=jackpot, line_dash="dash", line_color="gray",
                           annotation_text="Premio sin repartir")
            fig.update_layout(yaxis_title="COP si esa jugada gana", xaxis_title="")
            chart(fig, "Cuánto vale el premio mayor para cada jugada, si gana", "split_table")

            st.markdown("**Detalle por jugada**", help=HELP["split_table"])
            display = table.copy()
            display["Premio esperado (banda)"] = display.apply(
                lambda r: f"{r['expected_jackpot_share']:,.0f}  "
                          f"[{r['share_low']:,.0f} – {r['share_high']:,.0f}]", axis=1)
            # compare_tickets sorts its rows, so the percentile is joined by the
            # ticket's own string rather than by position.
            percentiles = {str(t): sum_percentile(t.main) for t in to_compare}
            display["Percentil de la suma"] = display["ticket"].map(percentiles)
            st.dataframe(
                display[["ticket", "popularity_score", "Percentil de la suma",
                         "expected_other_winners", "Premio esperado (banda)"]]
                .rename(columns={"ticket": "Jugada", "popularity_score": "Popularidad",
                                  "expected_other_winners": "Otros ganadores esperados"})
                .style.format({"Popularidad": "{:.3f}", "Otros ganadores esperados": "{:.3f}",
                                "Percentil de la suma": "{:.0%}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "**Percentil de la suma** es qué porcentaje de todas las combinaciones suma igual "
                "o menos que ésta. Cerca del 50% es una suma del montón, que es donde se concentra "
                "casi todo el mundo; cerca de 0% o 100% es una suma inusual. No cambia tu "
                "probabilidad de ganar ni un poco — es otra vista del mismo efecto de reparto. "
                "La distribución completa está en **5 · ¿Es aleatorio?**",
                help=HELP["sum_percentile"],
            )

            best, worst = table.iloc[0], table.iloc[-1]
            ratio = best["expected_jackpot_share"] / worst["expected_jackpot_share"]
            st.info(
                f"La menos popular de estas vale **{ratio:.2f}×** lo que vale la más popular, *si gana*. "
                f"Ambas ganan con la misma probabilidad: 1 en {total_combinations():,}. Elegir números "
                "no cambia eso, y este panel no dice lo contrario — solo cambia entre cuántos repartes."
            )
            if ratio < 1.15:
                st.caption(
                    f"Con {tickets_sold:,} tiquetes vendidos contra {total_combinations():,} "
                    "combinaciones, casi nunca hay con quién repartir, así que el efecto es pequeño. "
                    "Crece cuando se venden muchos más tiquetes — que es justo cuando el acumulado está "
                    "alto y más gente juega."
                )

            st.markdown("**Los sesgos por separado**", help=HELP["bias_components"])
            components = table[["ticket", *popularity_components(to_compare[0]).keys()]]
            st.dataframe(
                components.rename(columns={
                    "ticket": "Jugada", "calendar": "Fechas (≤31)", "low_numbers": "Números bajos",
                    "consecutive": "Seguidos", "arithmetic": "Espaciado regular",
                    "lucky_numbers": "De la suerte", "round_decade": "Misma decena",
                }).style.format({c: "{:.2f}" for c in
                                  ["Fechas (≤31)", "Números bajos", "Seguidos", "Espaciado regular",
                                   "De la suerte", "Misma decena"]}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "`unpopular` está registrada como estrategia y aparece en *Medir estrategias*. Ahí "
                "saldrá que **no** le gana al azar — resultado correcto y ajeno al punto: la tasa de "
                "aciertos es estructuralmente incapaz de ver lo que esta estrategia mejora."
            )


    # --------------------------------------------------------------- Backtest
    with tabs[7]:
        section("¿Algún modelo le gana al azar?", "tab_backtest")
        st.markdown(
            "Backtest *walk-forward*: cada modelo se entrena solo con datos anteriores al sorteo que "
            "intenta predecir y se compara contra lo que realmente salió. El número que importa no es "
            "'cuántos aciertos' sino **cuántos más que el azar puro** (calculado exactamente con la "
            "distribución hipergeométrica, sin simulación)."
        )

        experiment = st.radio(
            "¿Qué sorteos dejar fuera del entrenamiento?",
            ["Últimos N sorteos", "Corte por fecha (holdout)"],
            horizontal=True,
            help=HELP["experiment_choice"],
        )

        if experiment == "Últimos N sorteos":
            c1, c2, c3 = st.columns(3)
            max_windows = max(5, min(40, n_draws - MIN_TRAIN_FLOOR))
            n_windows = c1.slider("Ventanas (sorteos a evaluar)", 5, max_windows, min(15, max_windows),
                                  help=HELP["n_windows"])
            max_train = max(MIN_TRAIN_FLOOR, n_draws - 1)
            min_train = c2.slider("Mínimo de sorteos para entrenar", MIN_TRAIN_FLOOR, max_train,
                                  min(60, max_train), help=HELP["min_train"])
            include_prophet = c3.checkbox("Incluir Prophet", value=PROPHET_AVAILABLE,
                                           disabled=not PROPHET_AVAILABLE,
                                           help=HELP["include_prophet"] if PROPHET_AVAILABLE
                                           else HELP["prophet_missing"])
            # On when installed: the checkpoint is cached after the first run, so
            # the only cost that ever surprised anyone is paid once. A comparison
            # table that quietly leaves out an installed model is the bug this
            # project already fixed once, for Prophet.
            include_timesfm = c3.checkbox("Incluir TimesFM", value=TIMESFM_AVAILABLE,
                                          key="wf_timesfm",
                                          disabled=not TIMESFM_AVAILABLE,
                                          help=HELP["include_timesfm"] if TIMESFM_AVAILABLE
                                          else HELP["timesfm_missing"])

            start_idx, total = bt.window_bounds(n_draws, n_windows, min_train)
            if start_idx >= total:
                st.warning(
                    f"Con {n_draws} sorteos y un mínimo de {min_train} para entrenar no queda ninguna "
                    "ventana por evaluar. Baja el mínimo de entrenamiento o usa un histórico más largo."
                )

            if st.button("Ejecutar backtest"):
                with st.spinner("Corriendo backtest walk-forward..."):
                    try:
                        results = bt.run_all(position_series, n_columns, n_windows=n_windows,
                                              min_train=min_train, include_prophet=include_prophet,
                                              include_timesfm=include_timesfm)
                        st.session_state["backtest_summary"] = bt.summarize(results)
                        st.session_state.pop("holdout", None)
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.caption(
                "Entrena con todos los sorteos hasta la fecha de corte y predice los que vinieron "
                "después, que ya sabemos cómo salieron. Es el mismo experimento de arriba, pero el "
                "resultado es un sorteo concreto con una fecha, no un promedio."
            )
            first_date, last_date = df["ds"].min().date(), df["ds"].max().date()
            default_cutoff = (df["ds"].max() - pd.Timedelta(days=60)).date()
            c1, c2, c3 = st.columns(3)
            cutoff = c1.date_input(
                "Entrenar con datos hasta (inclusive)", value=max(default_cutoff, first_date),
                min_value=first_date, max_value=last_date, help=HELP["cutoff_date"],
            )
            mode_label = c2.radio(
                "Modo", ["Reentrenar en cada sorteo", "Entrenar una vez en el corte"],
                help=HELP["holdout_mode"],
            )
            mode = "expanding" if mode_label.startswith("Reentrenar") else "frozen"
            include_prophet = c3.checkbox("Incluir Prophet", value=PROPHET_AVAILABLE, key="holdout_prophet",
                                           disabled=not PROPHET_AVAILABLE,
                                           help=HELP["include_prophet"] if PROPHET_AVAILABLE
                                           else HELP["prophet_missing"])
            include_timesfm = c3.checkbox("Incluir TimesFM", value=TIMESFM_AVAILABLE,
                                          key="holdout_timesfm",
                                          disabled=not TIMESFM_AVAILABLE,
                                          help=HELP["include_timesfm"] if TIMESFM_AVAILABLE
                                          else HELP["timesfm_missing"])

            n_train_preview, n_holdout_preview = bt.cutoff_bounds(df["ds"], pd.Timestamp(cutoff))
            st.caption(
                f"Entrenaría con **{n_train_preview}** sorteos y predeciría **{n_holdout_preview}**."
            )
            # Expanding refits every model once per held-out draw, so the cost grows with
            # the horizon while frozen stays flat. Say so before the click, not after:
            # a five-minute spinner with no warning reads as a hung app.
            if mode == "expanding" and n_holdout_preview > 12:
                st.warning(
                    f"Reentrenar en cada sorteo significa ajustar todos los modelos {n_holdout_preview} "
                    "veces, así que esto puede tardar varios minutos (más aún con Prophet). "
                    "**Entrenar una vez en el corte** da un resultado en segundos con el mismo periodo "
                    "de prueba, o acerca la fecha de corte al final del histórico."
                )

            if st.button("Ejecutar holdout"):
                with st.spinner("Entrenando hasta el corte y prediciendo lo que ya pasó..."):
                    try:
                        results, info = bt.run_holdout(position_series, n_columns, pd.Timestamp(cutoff),
                                                        mode=mode, include_prophet=include_prophet,
                                                        include_timesfm=include_timesfm)
                        st.session_state["backtest_summary"] = bt.summarize(results)
                        st.session_state["holdout"] = (
                            bt.holdout_detail(results, position_series, n_columns), info
                        )
                    except ValueError as exc:
                        st.error(str(exc))

        if "holdout" in st.session_state:
            detail, info = st.session_state["holdout"]
            section("Sorteo por sorteo", "holdout_detail")
            st.caption(
                f"Entrenado con {info['n_train']} sorteos hasta {info['cutoff']:%Y-%m-%d}; "
                f"prediciendo {info['n_holdout']} sorteos entre {info['holdout_start']:%Y-%m-%d} y "
                f"{info['holdout_end']:%Y-%m-%d} "
                f"({'reentrenando en cada sorteo' if info['mode'] == 'expanding' else 'con un solo ajuste en el corte'})."
            )
            shown = detail.copy()
            shown["ds"] = shown["ds"].dt.strftime("%Y-%m-%d")
            st.dataframe(shown, use_container_width=True, hide_index=True)
            hit_columns = [c for c in detail.columns if c.endswith(" aciertos")]
            if hit_columns:
                fig = go.Figure()
                for column in hit_columns:
                    fig.add_scatter(x=detail["ds"], y=detail[column], mode="lines+markers",
                                    name=column.replace(" aciertos", ""))
                fig.add_hline(y=expected_main_matches(MAIN_BALLS_DRAWN)["mean"], line_dash="dash",
                              annotation_text="Azar esperado")
                fig.update_layout(yaxis_title="Aciertos (de 5)")
                chart(fig, "Aciertos por sorteo en el periodo de prueba", "holdout_chart")
            st.caption(
                "Un sorteo con 3 aciertos no es una señal: con 5 números elegidos de 43, acertar 3 o más "
                "pasa alrededor del 1% de las veces por puro azar, así que en una tabla con varios "
                "modelos y varios sorteos es normal ver alguno. Lo que decide es el promedio de abajo."
            )

        if "backtest_summary" in st.session_state:
            summary = st.session_state["backtest_summary"]
            winners = summary.loc[summary["beats_chance_corrected"], "model"].tolist()
            best = summary.loc[summary["avg_main_hits"].idxmax()]
            # good_is_pass=False: here "nothing beat chance" is the expected and
            # correct outcome, so it is the green one.
            plain_verdict(
                bool(winners),
                f"{', '.join(winners)} superó al azar — revísalo dos veces"
                if winners else
                "Ningún modelo le ganó al azar, que es el resultado correcto",
                f"El mejor fue **{best['model']}** con {best['avg_main_hits']:.2f} aciertos "
                f"promedio contra {best['chance_avg_main_hits']:.2f} del azar puro. "
                + ("Antes de creerlo: vuelve a correrlo con más ventanas y mira el intervalo de "
                   "confianza de abajo. Un modelo que gana en una corrida y no en la siguiente no "
                   "ganó."
                   if winners else
                   "Una diferencia pequeña en cualquier dirección es varianza, no habilidad."),
                good_is_pass=False,
            )
            fig = go.Figure()
            fig.add_bar(x=summary["model"], y=summary["avg_main_hits"], name="Modelo")
            fig.add_bar(x=summary["model"], y=summary["chance_avg_main_hits"], name="Azar (esperado)")
            fig.update_layout(barmode="group", yaxis_title="Aciertos promedio")
            chart(fig, "Aciertos promedio (balotas principales) vs. azar", "summary_chart")

            display = summary.copy()
            display["¿Le gana al azar? (p<0.05)"] = display["beats_chance"].map({True: "Sí", False: "No"})
            display["¿Le gana? (corregido)"] = display["beats_chance_corrected"].map({True: "Sí", False: "No"})
            display["Ventaja (IC 95%)"] = display.apply(
                lambda r: f"{r['effect']:+.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1)
            st.markdown("**Veredicto por modelo**", help=HELP["summary_table"])
            st.dataframe(
                display[["model", "n_windows", "avg_main_hits", "chance_avg_main_hits",
                         "Ventaja (IC 95%)", "p_value_better_than_chance",
                         "¿Le gana al azar? (p<0.05)",
                         "bonferroni_threshold", "¿Le gana? (corregido)", "super_hit_rate",
                         "chance_super_hit_rate"]]
                .style.format({
                    "avg_main_hits": "{:.2f}", "chance_avg_main_hits": "{:.2f}",
                    "p_value_better_than_chance": "{:.3f}", "bonferroni_threshold": "{:.4f}",
                    "super_hit_rate": "{:.3f}", "chance_super_hit_rate": "{:.3f}",
                }),
                use_container_width=True,
            )
            mde = minimum_detectable_effect(int(summary["n_windows"].max()))
            st.info(
                f"**Resolución de esta corrida.** Con {mde['n_draws']} sorteos evaluados, esta prueba solo "
                f"tiene 80% de probabilidad de detectar ventajas de **+{mde['relative']:.0%} o mayores** "
                f"(un promedio de {mde['detectable_mean']:.3f} aciertos contra {mde['chance_mean']:.3f} del "
                "azar). Un 'ningún modelo le gana al azar' aquí significa *ninguna ventaja mayor que eso* — "
                "no *ninguna ventaja*. Mira la pestaña **Potencia y Sensibilidad** para ver cuánta historia "
                "haría falta para afinar más."
            )
            st.caption(
                "La columna **Ventaja (IC 95%)** es la que dice cuánta resolución tuvo la corrida: un "
                "intervalo ancho que cruza el cero no es 'no hay ventaja', es 'no alcanzó a medirlo'.",
                help=HELP["effect_ci"],
            )
            st.caption(
                "El p-valor es de una cola: mide si el modelo es *mejor* que el azar, no solo distinto "
                "(un modelo peor que el azar no cuenta como que le gana). **Lee la columna corregida**: "
                f"se prueban {len(summary)} modelos contra los mismos sorteos, así que hay "
                f"{len(summary)} oportunidades de que alguno pase el 5% por suerte — la corrección de "
                "Bonferroni baja el umbral a 0.05 dividido entre el número de modelos. Con pocas "
                "ventanas, incluso un modelo sin señal real puede parecer mejor o peor por pura varianza."
            )


    # ------------------------------------------------ Potencia y Sensibilidad
    with tabs[8]:
        section("¿Qué podrían haber visto estas pruebas?", "tab_power")
        st.markdown(
            "Las otras pestañas responden *¿encontré una ventaja?*. Esta responde las dos preguntas que "
            "van antes, y sin las cuales un resultado nulo no significa nada: **¿podría haberla "
            "encontrado si existiera?** y **¿estas pruebas son capaces de detectar una ventaja real?**"
        )

        # ------------------------------------------------------------ potencia
        section("1. Efecto mínimo detectable", "mde_section")
        st.markdown(
            "Con la media del azar en 0.58 aciertos y una desviación estándar de 0.68, unos pocos "
            "cientos de sorteos solo alcanzan para revelar una ventaja bastante grande. Esto calcula "
            "exactamente cuál, para la misma prueba z que usan el backtest y las estrategias."
        )
        c1, c2, c3 = st.columns(3)
        mde_draws = c1.slider("Sorteos evaluados", 10, max(2000, n_draws), min(n_draws, 200), step=10,
                              help=HELP["mde_draws"])
        alpha = c2.select_slider("Nivel α", options=[0.01, 0.05, 0.10], value=0.05,
                                 help="Probabilidad de marcar una ventaja que no existe.")
        target_power = c3.select_slider("Potencia objetivo", options=[0.50, 0.80, 0.90, 0.95], value=0.80,
                                        help="Probabilidad de detectar la ventaja si sí existe.")

        mde = minimum_detectable_effect(mde_draws, alpha=alpha, power=target_power)
        m1, m2, m3 = st.columns(3)
        m1.metric("Ventaja mínima detectable", f"+{mde['relative']:.0%}", help=HELP["mde_metric"])
        m2.metric("Promedio que habría que alcanzar", f"{mde['detectable_mean']:.3f}",
                  help=HELP["mde_target"])
        m3.metric("Media del azar", f"{mde['chance_mean']:.3f}",
                  help="5 × 5 / 43. La referencia exacta, sin simulación.")
        st.caption(power_describe(mde_draws, alpha=alpha, power=target_power))

        curve = power_curve(mde_draws, alpha=alpha)
        fig = go.Figure()
        fig.add_scatter(x=curve["relative_edge"] * 100, y=curve["power"], mode="lines", name="Potencia")
        fig.add_hline(y=target_power, line_dash="dash", line_color="gray",
                      annotation_text=f"{target_power:.0%}")
        fig.update_layout(xaxis_title="Tamaño de la ventaja (% sobre el azar)", yaxis_title="Potencia",
                          yaxis_tickformat=".0%")
        chart(fig, f"Potencia con {mde_draws} sorteos", "power_chart")

        st.markdown("**Cuánta historia haría falta**", help=HELP["required_table"])
        needed = required_draws_table(alpha=alpha, power=target_power)
        st.dataframe(
            needed.rename(columns={
                "relative_edge": "Ventaja", "target_mean": "Promedio objetivo",
                "required_draws": "Sorteos necesarios", "years_of_history": "Años de historia",
            }).style.format({
                "Ventaja": "{:.0%}", "Promedio objetivo": "{:.4f}",
                "Sorteos necesarios": "{:,.0f}", "Años de historia": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
        super_mde = super_minimum_detectable_effect(mde_draws, alpha=alpha, power=target_power)
        st.caption(
            f"Superbalota: la tasa del azar es {super_mde['chance_rate']:.4f} (1 de 16) y la más pequeña "
            f"detectable con {mde_draws} sorteos es {super_mde['detectable_rate']:.4f} "
            f"(+{super_mde['relative']:.0%}).",
            help=HELP["super_mde"],
        )

        st.divider()

        # --------------------------------------------------------- sensibilidad
        section("2. ¿Detectan estas pruebas una ventaja real?", "sensitivity_section")
        st.markdown(
            "Genera sorteos con un sesgo **plantado a propósito** (3 números salen más de la cuenta) y "
            "mide cuántas veces lo detecta cada prueba. `pooled` mira los datos directamente; `hot` pone "
            "a prueba toda la cadena — generación, puntuación, línea base, prueba z; `random` es el "
            "control que **debe** quedarse en el piso incluso con sesgo, porque un tiquete uniforme no "
            "sabe cuáles números están favorecidos."
        )
        c1, c2, c3 = st.columns(3)
        sens_draws = c1.slider("Sorteos por experimento", 100, 1000, 500, step=100)
        sens_seeds = c2.slider("Semillas", 5, 40, 10, help=HELP["sensitivity_seeds"])
        detectors = c3.multiselect("Detectores", ["pooled", "hot", "random"],
                                   default=["pooled", "hot", "random"])
        strengths = st.multiselect(
            "Fuerzas del sesgo a probar", list(DEFAULT_STRENGTHS) + [3.0, 4.0],
            default=list(DEFAULT_STRENGTHS), help=HELP["sensitivity_strengths"])

        slow = [d for d in detectors if d in ("hot", "random")]
        if slow and sens_seeds * len(strengths) * len(slow) > 100:
            st.warning(
                f"{len(strengths)} fuerzas × {sens_seeds} semillas × {len(slow)} detector(es) de "
                "estrategia significa regenerar datos y tiquetes muchas veces: esto puede tardar varios "
                "minutos. Quita `hot`/`random`, o baja las semillas, para una vista rápida — `pooled` "
                "solo es casi instantáneo."
            )

        if st.button("Ejecutar prueba de sensibilidad"):
            if not detectors or not strengths:
                st.error("Elige al menos un detector y una fuerza de sesgo.")
            elif 0.0 not in strengths:
                st.error(
                    "Incluye la fuerza **0.0**: es el control. Sin ella no puedes saber si una tasa de "
                    "detección alta significa sensibilidad o un detector roto."
                )
            else:
                with st.spinner("Generando datos sesgados y corriendo los detectores..."):
                    st.session_state["sensitivity"] = sensitivity_report(
                        strengths=tuple(sorted(strengths)), detectors=tuple(detectors),
                        n_draws=sens_draws, n_seeds=sens_seeds, alpha=alpha,
                        n_draws_back=min(200, sens_draws - MIN_TRAIN_FLOOR), tickets_per_draw=5,
                    )

        if "sensitivity" in st.session_state:
            report = st.session_state["sensitivity"]

            fig = go.Figure()
            for detector, group in report.groupby("detector"):
                group = group.sort_values("strength")
                fig.add_scatter(x=group["strength"], y=group["detection_rate"], mode="lines+markers",
                                name=detector)
            fig.add_hline(y=0.80, line_dash="dash", line_color="gray", annotation_text="80%")
            fig.add_hline(y=alpha, line_dash="dot", line_color="gray",
                          annotation_text=f"α = {alpha:g}")
            fig.update_layout(xaxis_title="Fuerza del sesgo inyectado", yaxis_title="Tasa de detección",
                              yaxis_tickformat=".0%")
            chart(fig, "¿Cuánto sesgo hace falta para que cada prueba lo vea?", "sensitivity_chart")

            st.markdown("**Tasas de detección**", help=HELP["sensitivity_table"])
            st.dataframe(
                report.rename(columns={
                    "detector": "Detector", "strength": "Fuerza",
                    "favored_share": "% de balotas que se llevaron", "uniform_share": "% si fuera uniforme",
                    "times_detected": "Detectado", "n_seeds": "Semillas",
                    "detection_rate": "Tasa", "median_p_value": "p mediano",
                }).style.format({
                    "% de balotas que se llevaron": "{:.2%}", "% si fuera uniforme": "{:.2%}",
                    "Tasa": "{:.0%}", "p mediano": "{:.4f}",
                }),
                use_container_width=True, hide_index=True,
            )

            control = report[report["strength"] == 0.0]
            broken = control[control["detection_rate"] > 4 * alpha]
            if not broken.empty:
                st.warning(
                    "Sin sesgo, "
                    + ", ".join(f"`{r.detector}` disparó {r.detection_rate:.0%}" for r in broken.itertuples())
                    + f", muy por encima de α = {alpha:g}. Con pocas semillas esto puede ser ruido, pero "
                    "si se sostiene al subirlas, ese detector marca ventajas que no existen y sus "
                    "resultados positivos no son confiables."
                )
            else:
                st.success(
                    f"Control sano: sin sesgo, ningún detector supera α = {alpha:g} de forma preocupante. "
                    "Las filas con sesgo sí miden sensibilidad."
                )

            for detector in report["detector"].unique():
                threshold = sensitivity_threshold(report, detector)
                if threshold is None:
                    st.error(
                        f"**{detector}** nunca llegó al 80% de detección en el rango probado. Un resultado "
                        "nulo suyo no descarta nada más pequeño que el sesgo más fuerte que probaste."
                    )
                else:
                    st.info(
                        f"**{detector}** alcanza el 80% de detección con fuerza {threshold['strength']:g}, "
                        f"donde los 3 números favorecidos se llevan el {threshold['favored_share']:.2%} de "
                        f"las balotas en vez del {threshold['uniform_share']:.2%} uniforme. Ese es su "
                        "umbral de sensibilidad: por debajo, no puede ver nada."
                    )


    # ----------------------------------------------------------------- Registro
    with tabs[9]:
        section("Predicciones registradas antes del sorteo", "tab_registry")
        st.markdown(
            "Todo lo demás en este panel es **retrospectivo**, y el análisis retrospectivo siempre se "
            "puede ajustar después: una ventana corrida, un modelo cambiado, una corrida que no se "
            "cuenta. Nada de eso es deshonestidad — es lo que le pasa a cualquiera que analiza datos "
            "que ya vio. Lo único que no se puede ajustar después es una predicción escrita **antes** "
            "de que el resultado existiera. Eso es todo lo que hace esta pestaña."
        )
        if is_demo:
            st.warning(
                "Estás sobre datos sintéticos de demostración. Puedes registrar predicciones, pero se "
                "puntuarán contra sorteos inventados. Carga tu CSV real antes de empezar un registro "
                "que quieras tomar en serio."
            )

        state = registry_status()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registradas", state["n_recorded"])
        m2.metric("Puntuadas", state["n_scored"])
        m3.metric("Pendientes", state["n_pending"], help=HELP["registry_pending"])
        m4.metric("Ventaja mínima detectable",
                  f"+{state['min_detectable_effect']:.0%}" if state["n_scored"] else "—",
                  help=HELP["registry_mde"])

        # ------------------------------------------------------------- registrar
        section("Registrar una jugada", "registry_record")
        weekdays = infer_draw_weekdays(df["ds"])
        upcoming = next_draw_dates(max(df["ds"].max(), pd.Timestamp.today().normalize()), 6,
                                   weekdays=weekdays)
        c1, c2 = st.columns([2, 1])
        reg_main = c1.text_input(
            f"5 balotas ({MAIN_BALL_RANGE[0]}-{MAIN_BALL_RANGE[1]}), separadas por coma o guion",
            "3, 12, 19, 27, 41", key="reg_main")
        reg_super = c2.number_input("Superbalota", min_value=SUPER_BALL_RANGE[0],
                                    max_value=SUPER_BALL_RANGE[1], value=8, key="reg_super")
        c1, c2, c3 = st.columns(3)
        reg_date = c1.selectbox("Sorteo", upcoming, format_func=lambda d: f"{d:%Y-%m-%d (%a)}",
                                help="Solo sorteos futuros: el registro rechaza cualquier otro.")
        reg_label = c2.text_input("Etiqueta", "yo", help=HELP["registry_label"])
        reg_note = c3.text_input("Nota (opcional)", "")

        if st.button("Registrar predicción"):
            try:
                ticket = Ticket(main=tuple(int(x) for x in reg_main.replace("-", ",").split(",")
                                           if x.strip()),
                                super_ball=int(reg_super))
                row = record_prediction(ticket, reg_date, reg_label.strip() or "sin etiqueta",
                                        note=reg_note)
            except (ValueError, TypeError) as exc:
                st.error(f"Jugada inválida: {exc}")
            except RegistryError as exc:
                # The module raises in English (it is library code); this is the end-user
                # product, so the reason gets said in Spanish and the raw detail goes
                # underneath rather than being dropped.
                if "already on the record" in str(exc):
                    st.error(
                        f"Ya hay una predicción con la etiqueta `{reg_label}` para el sorteo del "
                        f"{reg_date:%Y-%m-%d}. El registro es solo-anexar: no se sobreescribe nada. "
                        "Si cambiaste de opinión, regístrala con otra etiqueta y quedan las dos — que "
                        "es precisamente el punto."
                    )
                else:
                    st.error(
                        f"Ese sorteo ({reg_date:%Y-%m-%d}) no está en el futuro. Una predicción "
                        "anotada después de su sorteo no demuestra nada, así que el registro no la "
                        "acepta: cada fila del archivo tiene que haber sido falsable cuando se escribió."
                    )
                st.caption(f"Detalle: {exc}")
            else:
                st.success(
                    f"Registrada **{row['main']} + {row['super_ball']}** para el sorteo del "
                    f"{row['draw_date']:%Y-%m-%d}, como `{row['label']}`, a las {row['recorded_at']}. "
                    "Ya no se puede cambiar — que es justamente lo que la vuelve evidencia."
                )
                st.rerun()

        # -------------------------------------------------------------- puntuar
        st.divider()
        c1, c2 = st.columns([1, 3])
        if c1.button("Puntuar sorteos ya ocurridos"):
            score_pending(df, balls_expanded)
            st.rerun()
        c2.caption(
            "Puntúa **todas** las predicciones cuyo sorteo ya salió, no un subconjunto: elegir cuáles "
            "contar es exactamente el sesgo que este registro existe para evitar. Volver a correrlo es "
            "seguro, las filas ya puntuadas no se tocan."
        )

        registry = load_registry()
        if registry.empty:
            st.info(
                "El registro está vacío. Regístra una jugada arriba y vuelve después del sorteo: hasta "
                "que una predicción sobreviva a un sorteo que no habías visto, este registro no "
                "demuestra nada — y eso es correcto."
            )
        else:
            upcoming_rows = pending_predictions(registry=registry)
            if not upcoming_rows.empty:
                st.markdown("**Pendientes**", help=HELP["registry_pending"])
                shown = upcoming_rows[["draw_date", "label", "main", "super_ball", "recorded_at", "note"]].copy()
                shown["draw_date"] = pd.to_datetime(shown["draw_date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(shown, use_container_width=True, hide_index=True)

            table = registry_summary(registry=registry, by_label=True)
            if table.empty:
                st.caption("Nada puntuado todavía.")
            else:
                display = table.copy()
                display["Ventaja (IC 95%)"] = display.apply(
                    lambda r: f"{r['effect']:+.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", axis=1)
                st.markdown("**Resultado por etiqueta**", help=HELP["registry_summary"])
                st.dataframe(
                    display[["label", "n_scored", "avg_main_matches", "chance_avg_main_matches",
                             "Ventaja (IC 95%)", "p_value_better_than_chance", "min_detectable_effect",
                             "super_hit_rate", "chance_super_hit_rate"]]
                    .style.format({
                        "avg_main_matches": "{:.3f}", "chance_avg_main_matches": "{:.3f}",
                        "p_value_better_than_chance": "{:.3f}", "min_detectable_effect": "{:.0%}",
                        "super_hit_rate": "{:.3f}", "chance_super_hit_rate": "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )
                worst = table["min_detectable_effect"].min()
                st.caption(
                    f"Con las predicciones puntuadas hasta ahora, la ventaja más pequeña que este "
                    f"registro podría revelar es **+{worst:.0%}**. Por debajo de eso, un 'no le gana al "
                    "azar' habla del tamaño de la muestra, no de las predicciones.",
                    help=HELP["registry_mde"],
                )

                scored = registry[registry["main_matches"].notna()].copy()
                scored["draw_date"] = pd.to_datetime(scored["draw_date"])
                fig = go.Figure()
                for label, group in scored.groupby("label"):
                    group = group.sort_values("draw_date")
                    fig.add_scatter(x=group["draw_date"], y=group["main_matches"].astype(float),
                                    mode="lines+markers", name=str(label))
                fig.add_hline(y=expected_main_matches(MAIN_BALLS_DRAWN)["mean"], line_dash="dash",
                              annotation_text="Azar esperado")
                fig.update_layout(yaxis_title="Aciertos (de 5)")
                chart(fig, "Aciertos por sorteo registrado", "holdout_chart")

            with st.expander("Ver el registro completo"):
                st.caption(
                    "Se guarda en `predictions.csv`, en la raíz del repo y **no** ignorado por git. "
                    "Commitearlo pone cada predicción bajo control de versiones con una fecha encima, "
                    "que es un respaldo más fuerte que cualquier columna de timestamp que el propio "
                    "archivo se escriba."
                )
                st.dataframe(registry, use_container_width=True, hide_index=True)
