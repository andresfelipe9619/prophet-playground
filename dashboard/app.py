"""Baloto analytics dashboard.

Run with: streamlit run dashboard/app.py

The goal of this dashboard is decision support, not "the number to play":
every tab that touches a model or a heuristic ("hot numbers", "overdue
numbers", forecasts) is paired with the statistical check for whether that
signal is distinguishable from pure chance. Baloto draws are independent and
uniform by design, so the honest expectation is that most of these checks
come back negative — the dashboard is built to show that clearly instead of
hiding it.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest as bt
from analysis.randomness import (
    autocorrelation_check,
    frequency_table,
    gap_table,
    hot_cold_numbers,
    is_sorted_ascending,
    pooled_uniformity_test,
    randomness_report,
)
from analysis.prizes import (
    breakeven_jackpot,
    category_probabilities,
    expected_value,
    total_combinations,
)
from models.baseline import most_frequent_pick
from models.common import (
    DEFAULT_DATA_PATH,
    MAIN_BALLS_DRAWN,
    MAIN_BALL_RANGE,
    SUPER_BALL_RANGE,
    build_position_series,
    infer_draw_weekdays,
    main_positions,
    next_draw_dates,
    series_label,
    super_position,
)
from models.statsforecast_model import MODEL_NAMES, adjusted_predictions, fit_predict_all
from models.xgboost_model import forecast_next
from utils.processor import load_and_preprocess, preprocess_draws
from utils.sample_data import load_sample_and_preprocess

st.set_page_config(page_title="Baloto Analytics", layout="wide")

MIN_TRAIN_FLOOR = 20  # below this the models have nothing to learn from


@st.cache_data(show_spinner=False)
def load_data(path, uploaded_bytes):
    """Load draws and derive everything downstream needs, in one cached step.

    position_series comes back from here rather than from a second cached
    function so Streamlit never has to hash the full frames as arguments on
    every rerun.
    """
    if uploaded_bytes is not None:
        df, balls_expanded = preprocess_draws(pd.read_csv(io.BytesIO(uploaded_bytes)))
        is_demo = False
    elif os.path.exists(path):
        df, balls_expanded = load_and_preprocess(path)
        is_demo = False
    else:
        df, balls_expanded = load_sample_and_preprocess(n_draws=400)
        is_demo = True
    return df, balls_expanded, build_position_series(df, balls_expanded), is_demo


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


def verdict_badge(looks_random, positive_text="Sin evidencia de patrón explotable", negative_text="Posible señal — revisar"):
    if looks_random:
        st.success(positive_text)
    else:
        st.warning(negative_text)


st.title("🎯 Baloto Analytics")
st.caption(
    "Panel de análisis y forecasting para Baloto. Antes de leer cualquier predicción, revisa la pestaña "
    "**Aleatoriedad**: los sorteos de lotería son, por diseño, independientes y uniformes — el objetivo de este "
    "panel es mostrar honestamente si hay o no señal explotable, no prometer que la hay."
)

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("CSV propio (columnas Date, Ball)", type="csv")
    data_path = st.text_input("Ruta local (si no subes archivo)", value=DEFAULT_DATA_PATH)

df, balls_expanded, position_series, is_demo = load_data(
    data_path, uploaded.getvalue() if uploaded else None
)
n_columns = balls_expanded.shape[1]
n_draws = len(df)
label_to_pos = {series_label(p, n_columns): p for p in range(n_columns)}

if is_demo:
    st.info(
        "No se encontró un CSV real en `exported_data/` ni se subió uno propio — mostrando **datos sintéticos "
        "de demostración** (sorteos uniformes independientes generados aleatoriamente), solo para que puedas "
        "explorar el panel. Sube tu CSV real en la barra lateral para analizar tus datos."
    )

tabs = st.tabs([
    "Resumen", "Probabilidades y Valor Esperado", "Frecuencia y Gaps", "Hot / Cold",
    "Aleatoriedad", "Forecast", "Backtest vs. Azar",
])

# ---------------------------------------------------------------- Resumen
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sorteos", n_draws)
    col2.metric("Desde", df["ds"].min().strftime("%Y-%m-%d"))
    col3.metric("Hasta", df["ds"].max().strftime("%Y-%m-%d"))
    pooled_main, pooled_super, sorted_flag = cached_pooled_tests(balls_expanded, n_columns)
    col4.metric("Balotas guardadas ordenadas asc.", "Sí" if sorted_flag else "No")

    if sorted_flag:
        st.warning(
            "Tus datos parecen tener las 5 balotas principales guardadas de menor a mayor por sorteo. Eso "
            "convierte cada columna en un **estadístico de orden** (mínimo, 2do menor, ...), no en una balota "
            "uniforme — un chi-cuadrado por posición puede marcar 'no aleatorio' solo por el orden, no porque "
            "haya un patrón real. Usa la prueba agrupada (pooled) de abajo, que es inmune a esto."
        )

    st.subheader(
        f"¿Los números principales ({MAIN_BALL_RANGE[0]}-{MAIN_BALL_RANGE[1]}) se reparten uniformemente?"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.metric("p-valor (balotas principales, agrupadas)", f"{pooled_main['p_value']:.3f}")
        verdict_badge(pooled_main["p_value"] > 0.05)
    with c2:
        st.metric("p-valor (superbalota)", f"{pooled_super['p_value']:.3f}")
        verdict_badge(pooled_super["p_value"] > 0.05)
    st.caption(
        "p-valor alto (>0.05) = no hay evidencia contra la hipótesis de uniformidad, que es justamente lo "
        "esperable en un sorteo justo. Un p-valor bajo aquí sí sería una señal real y rara — vale la pena "
        "revisar la fuente de datos si eso pasa."
    )

# ------------------------------------------- Probabilidades y Valor Esperado
with tabs[1]:
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
    c1.metric("Probabilidad del premio mayor", f"1 en {jackpot_odds:,.0f}")
    c2.metric("Combinaciones posibles", f"{total_combinations():,}")

    st.subheader("Tabla de premios")
    st.caption(
        "Los montos de abajo son **valores de ejemplo que debes reemplazar** con la tabla oficial vigente "
        "(varias categorías son variables y el premio mayor se acumula). Pon 0 en las categorías que no "
        "pagan premio. Las probabilidades sí son exactas y no dependen de lo que escribas aquí."
    )

    ticket_price = st.number_input("Precio del tiquete (COP)", min_value=0, value=5700, step=100)

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

    st.subheader("Valor esperado por tiquete")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retorno esperado", f"${ev['expected_return']:,.0f}")
    m2.metric("Valor esperado", f"${ev['expected_value']:,.0f}",
              delta=f"{ev['expected_value']:,.0f} por tiquete")
    m3.metric("Retorno al jugador (RTP)", f"{ev['return_to_player'] * 100:.1f}%")
    m4.metric("Prob. de ganar algo", f"1 en {ev['odds_any_prize_one_in']:,.1f}")

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
    st.metric("Premio mayor necesario para que el valor esperado sea cero", f"${breakeven:,.0f}")

    st.subheader("Probabilidad por categoría")
    plot_df = prob_table[prob_table["probability"] > 0].copy()
    fig = go.Figure()
    fig.add_bar(x=plot_df["category"], y=plot_df["probability"])
    fig.update_layout(
        yaxis_type="log", yaxis_title="Probabilidad (escala log)", xaxis_title="Categoría",
        title="Probabilidad exacta de cada categoría de premio",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------- Frecuencia y Gaps
with tabs[2]:
    chosen_label = st.selectbox("Posición", list(label_to_pos.keys()), key="freq_pos")
    pos = label_to_pos[chosen_label]

    freq = frequency_table(position_series[pos], pos, n_columns)
    fig = go.Figure()
    fig.add_bar(x=freq["number"], y=freq["count"], name="Observado")
    fig.add_hline(y=float(freq["expected_count"].iloc[0]), line_dash="dash",
                  annotation_text="Esperado (uniforme)", line_color="gray")
    fig.update_layout(title=f"Frecuencia — {chosen_label}", xaxis_title="Número", yaxis_title="Veces salido")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gaps entre apariciones y 'número atrasado'")
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
    chosen_label_hc = st.selectbox("Posición", list(label_to_pos.keys()), key="hc_pos")
    pos_hc = label_to_pos[chosen_label_hc]
    window = st.slider("Ventana reciente (# sorteos)", 5, 60, 20)

    hc = hot_cold_numbers(position_series[pos_hc], pos_hc, n_columns, recent_draws=window)
    fig = go.Figure()
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in hc["delta_pct"]]
    fig.add_bar(x=hc["number"], y=hc["delta_pct"], marker_color=colors)
    fig.update_layout(
        title=f"Hot (rojo) / Cold (azul) — {chosen_label_hc} (últimos {window} sorteos vs. histórico)",
        xaxis_title="Número", yaxis_title="Diferencia de participación (%)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Diferencia entre el % de apariciones en la ventana reciente y el % histórico. Con pocas observaciones "
        "por ventana, este ruido es esperable incluso sin ningún cambio real en el proceso de sorteo."
    )

# ------------------------------------------------------------ Aleatoriedad
with tabs[4]:
    st.subheader("Veredicto por posición")
    reports = randomness_reports(position_series, n_columns)
    rows = []
    for p, rep in reports.items():
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

    st.subheader("Autocorrelación (ACF)")
    acf_label = st.selectbox("Posición", list(label_to_pos.keys()), key="acf_pos")
    acf_pos = label_to_pos[acf_label]
    autocorr = reports[acf_pos]["autocorrelation"]  # already computed above
    n = len(position_series[acf_pos])
    band = 1.96 / (n ** 0.5)
    fig = go.Figure()
    fig.add_bar(x=list(range(len(autocorr["acf"]))), y=autocorr["acf"], name="ACF")
    fig.add_hline(y=band, line_dash="dash", line_color="gray")
    fig.add_hline(y=-band, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"ACF — {acf_label}", xaxis_title="Lag", yaxis_title="Autocorrelación")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Ljung-Box p-valor (¿hay autocorrelación?)", f"{autocorr['ljung_box_p_value']:.3f}")
    verdict_badge(autocorr["ljung_box_p_value"] > 0.05,
                  "Sin autocorrelación detectable — no hay 'memoria' que un modelo de series de tiempo pueda explotar",
                  "Autocorrelación detectada — esto sí justificaría probar un modelo de series de tiempo")

# ------------------------------------------------------------------ Forecast
with tabs[5]:
    st.warning(
        "Estos son ejercicios de forecasting, no predicciones confiables: para un sorteo justo, ningún modelo "
        "puede superar de forma sostenida la probabilidad teórica. Revisa la pestaña Backtest antes de confiar "
        "en cualquiera de estos números."
    )
    model_choice = st.selectbox("Modelo", ["FrequencyBaseline", "Prophet", *MODEL_NAMES, "XGBoost"])

    if st.button("Generar predicción del próximo sorteo"):
        with st.spinner("Entrenando..."):
            history = position_series[0]["ds"]
            next_date = next_draw_dates(history.max(), 1, weekdays=infer_draw_weekdays(history))[0]

            preds = {}
            if model_choice == "FrequencyBaseline":
                preds = most_frequent_pick(position_series)
            elif model_choice == "Prophet":
                from Prophet import define_and_fit_model, predict_at_dates
                for p in range(n_columns):
                    m = define_and_fit_model(position_series[p])  # sin festivos: no afectan una balota
                    # predict_at_dates evalúa solo la fecha pedida; make_predictions
                    # re-predeciría toda la historia para usar una sola fila.
                    fc = predict_at_dates(m, p, n_columns, [next_date])
                    preds[p] = int(fc["yhat_adjusted"].iloc[0])
            elif model_choice in MODEL_NAMES:
                raw = fit_predict_all(position_series, h=1)
                clipped = adjusted_predictions(raw, n_columns, model_name=model_choice)
                preds = dict(zip(clipped["unique_id"].astype(int), clipped["yhat_adjusted"].astype(int)))
            elif model_choice == "XGBoost":
                for p in range(n_columns):
                    preds[p] = forecast_next(position_series[p], p, n_columns, next_date)

        if any(preds.get(p) is None for p in range(n_columns)):
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

# --------------------------------------------------------------- Backtest
with tabs[6]:
    st.markdown(
        "Backtest *walk-forward*: en cada sorteo histórico reciente, cada modelo se entrena solo con datos "
        "anteriores a ese sorteo y se compara contra lo que realmente salió. El número que importa no es "
        "'cuántos aciertos' sino **cuántos más que el azar puro** (calculado exactamente con la distribución "
        "hipergeométrica, sin simulación)."
    )
    c1, c2, c3 = st.columns(3)
    max_windows = max(5, min(40, n_draws - MIN_TRAIN_FLOOR))
    n_windows = c1.slider("Ventanas (sorteos a evaluar)", 5, max_windows, min(15, max_windows))
    max_train = max(MIN_TRAIN_FLOOR, n_draws - 1)
    min_train = c2.slider("Mínimo de sorteos para entrenar", MIN_TRAIN_FLOOR, max_train,
                          min(60, max_train))
    include_prophet = c3.checkbox("Incluir Prophet (más lento)", value=False)

    start, total = bt.window_bounds(n_draws, n_windows, min_train)
    if start >= total:
        st.warning(
            f"Con {n_draws} sorteos y un mínimo de {min_train} para entrenar no queda ninguna ventana "
            "por evaluar. Baja el mínimo de entrenamiento o usa un histórico más largo."
        )

    if st.button("Ejecutar backtest"):
        with st.spinner("Corriendo backtest walk-forward..."):
            try:
                results = bt.run_all(position_series, n_columns, n_windows=n_windows,
                                      min_train=min_train, include_prophet=include_prophet)
                st.session_state["backtest_summary"] = bt.summarize(results)
            except ValueError as exc:
                st.error(str(exc))

    if "backtest_summary" in st.session_state:
        summary = st.session_state["backtest_summary"]
        fig = go.Figure()
        fig.add_bar(x=summary["model"], y=summary["avg_main_hits"], name="Modelo")
        fig.add_bar(x=summary["model"], y=summary["chance_avg_main_hits"], name="Azar (esperado)")
        fig.update_layout(barmode="group", title="Aciertos promedio (balotas principales) vs. azar",
                           yaxis_title="Aciertos promedio")
        st.plotly_chart(fig, use_container_width=True)

        display = summary.copy()
        display["¿Le gana al azar? (p<0.05)"] = display["beats_chance_p<0.05"].map({True: "Sí", False: "No"})
        st.dataframe(
            display[["model", "n_windows", "avg_main_hits", "chance_avg_main_hits",
                     "p_value_better_than_chance", "¿Le gana al azar? (p<0.05)", "super_hit_rate",
                     "chance_super_hit_rate"]]
            .style.format({
                "avg_main_hits": "{:.2f}", "chance_avg_main_hits": "{:.2f}",
                "p_value_better_than_chance": "{:.3f}",
                "super_hit_rate": "{:.3f}", "chance_super_hit_rate": "{:.3f}",
            }),
            use_container_width=True,
        )
        st.caption(
            "El p-valor es de una cola: mide si el modelo es *mejor* que el azar, no solo distinto (un modelo "
            "peor que el azar no cuenta como que le gana). Con pocas ventanas, incluso un modelo sin señal real "
            "puede parecer mejor o peor por pura varianza — desconfía de una sola corrida con pocas ventanas."
        )
