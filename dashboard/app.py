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
from contants import COLOMBIA_HOLIDAYS
from models.baseline import beats_chance_test, empirical_frequency_pick, expected_super_match_rate
from models.common import build_position_series, series_label
from models.statsforecast_model import MODEL_NAMES, adjusted_predictions, fit_predict_all
from models.xgboost_model import train_predict
from utils.processor import load_and_preprocess
from utils.sample_data import load_sample_and_preprocess

st.set_page_config(page_title="Baloto Analytics", layout="wide")

DEFAULT_DATA_PATH = "exported_data/final-final.csv"
MAIN_POSITIONS = 5


@st.cache_data(show_spinner=False)
def load_data(path, uploaded_bytes):
    if uploaded_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
        df["ds"] = pd.to_datetime(df["Date"], dayfirst=True)
        balls_expanded = df["Ball"].str.split("-", expand=True).apply(pd.to_numeric)
        return df, balls_expanded, False
    if os.path.exists(path):
        df, balls_expanded = load_and_preprocess(path)
        return df, balls_expanded, False
    df, balls_expanded = load_sample_and_preprocess(n_draws=400)
    return df, balls_expanded, True


@st.cache_data(show_spinner=False)
def compute_position_series(df, balls_expanded):
    return build_position_series(df, balls_expanded)


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

df, balls_expanded, is_demo = load_data(data_path, uploaded.getvalue() if uploaded else None)
n_columns = balls_expanded.shape[1]
position_series = compute_position_series(df, balls_expanded)
n_draws = len(df)

if is_demo:
    st.info(
        "No se encontró un CSV real en `exported_data/` ni se subió uno propio — mostrando **datos sintéticos "
        "de demostración** (sorteos uniformes independientes generados aleatoriamente), solo para que puedas "
        "explorar el panel. Sube tu CSV real en la barra lateral para analizar tus datos."
    )

tabs = st.tabs(["Resumen", "Frecuencia y Gaps", "Hot / Cold", "Aleatoriedad", "Forecast", "Backtest vs. Azar"])

# ---------------------------------------------------------------- Resumen
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sorteos", n_draws)
    col2.metric("Desde", df["ds"].min().strftime("%Y-%m-%d"))
    col3.metric("Hasta", df["ds"].max().strftime("%Y-%m-%d"))
    sorted_flag = is_sorted_ascending(balls_expanded, main_positions=MAIN_POSITIONS)
    col4.metric("Balotas guardadas ordenadas asc.", "Sí" if sorted_flag else "No")

    if sorted_flag:
        st.warning(
            "Tus datos parecen tener las 5 balotas principales guardadas de menor a mayor por sorteo. Eso "
            "convierte cada columna en un **estadístico de orden** (mínimo, 2do menor, ...), no en una balota "
            "uniforme — un chi-cuadrado por posición puede marcar 'no aleatorio' solo por el orden, no porque "
            "haya un patrón real. Usa la prueba agrupada (pooled) de abajo, que es inmune a esto."
        )

    st.subheader("¿Los números principales (1-43) se reparten uniformemente?")
    pooled_main = pooled_uniformity_test(balls_expanded, 1, 43, positions=list(range(MAIN_POSITIONS)))
    pooled_super = pooled_uniformity_test(balls_expanded, 1, 16, positions=[n_columns - 1])
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

# ---------------------------------------------------------- Frecuencia y Gaps
with tabs[1]:
    label_to_pos = {series_label(p, n_columns): p for p in range(n_columns)}
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
    gaps = gap_table(position_series[pos], pos, n_columns)
    st.dataframe(
        gaps.sort_values("overdue_score", ascending=False)
        .style.format({"avg_gap_days": "{:.1f}", "std_gap_days": "{:.1f}", "overdue_score": "{:.2f}"}),
        use_container_width=True,
    )

# ---------------------------------------------------------------- Hot/Cold
with tabs[2]:
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
with tabs[3]:
    st.subheader("Veredicto por posición")
    rows = []
    for p in range(n_columns):
        rep = randomness_report(position_series[p], p, n_columns)
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
    autocorr = autocorrelation_check(position_series[acf_pos]["y"].to_numpy())
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
with tabs[4]:
    st.warning(
        "Estos son ejercicios de forecasting, no predicciones confiables: para un sorteo justo, ningún modelo "
        "puede superar de forma sostenida la probabilidad teórica. Revisa la pestaña Backtest antes de confiar "
        "en cualquiera de estos números."
    )
    model_choice = st.selectbox("Modelo", ["FrequencyBaseline", "Prophet", *MODEL_NAMES, "XGBoost"])

    if st.button("Generar predicción del próximo sorteo"):
        with st.spinner("Entrenando..."):
            preds = {}
            if model_choice == "FrequencyBaseline":
                for p in range(n_columns):
                    preds[p] = int(position_series[p]["y"].mode().iloc[0])
            elif model_choice == "Prophet":
                from Prophet import define_and_fit_model, make_predictions
                for p in range(n_columns):
                    m = define_and_fit_model(position_series[p], holidays=COLOMBIA_HOLIDAYS)
                    fc = make_predictions(m, p, n_columns, periods=1)
                    preds[p] = int(fc["yhat_adjusted"].iloc[-1])
            elif model_choice in MODEL_NAMES:
                raw = fit_predict_all(position_series, h=1)
                clipped = adjusted_predictions(raw, n_columns, model_name=model_choice)
                for p in range(n_columns):
                    preds[p] = int(clipped.loc[clipped["unique_id"] == p, "yhat_adjusted"].iloc[0])
            elif model_choice == "XGBoost":
                for p in range(n_columns):
                    result = train_predict(position_series[p], p, n_columns, test_size=0.05)
                    preds[p] = int(result["test"]["yhat_adjusted"].iloc[-1])

        main_numbers = sorted({preds[p] for p in range(MAIN_POSITIONS)})
        super_number = preds[n_columns - 1]
        st.success(f"Balotas principales sugeridas: **{' - '.join(str(n) for n in main_numbers)}**  |  "
                   f"Superbalota: **{super_number}**")
        if len(main_numbers) < MAIN_POSITIONS:
            st.caption(f"Nota: hubo {MAIN_POSITIONS - len(main_numbers)} coincidencia(s) entre posiciones, "
                       "por eso hay menos de 5 números distintos — típico cuando el modelo no tiene señal real "
                       "que diferencie una posición de otra.")

# --------------------------------------------------------------- Backtest
with tabs[5]:
    st.markdown(
        "Backtest *walk-forward*: en cada sorteo histórico reciente, cada modelo se entrena solo con datos "
        "anteriores a ese sorteo y se compara contra lo que realmente salió. El número que importa no es "
        "'cuántos aciertos' sino **cuántos más que el azar puro** (calculado exactamente con la distribución "
        "hipergeométrica, sin simulación)."
    )
    c1, c2, c3 = st.columns(3)
    n_windows = c1.slider("Ventanas (sorteos a evaluar)", 5, 40, 15)
    min_train = c2.slider("Mínimo de sorteos para entrenar", 20, max(21, n_draws - n_windows - 1), 60)
    include_prophet = c3.checkbox("Incluir Prophet (más lento)", value=False)

    if st.button("Ejecutar backtest"):
        with st.spinner("Corriendo backtest walk-forward..."):
            results = bt.run_all(position_series, n_columns, n_windows=n_windows,
                                  min_train=min_train, include_prophet=include_prophet)
            summary = bt.summarize(results)
        st.session_state["backtest_summary"] = summary

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
            display[["model", "n_windows", "avg_main_hits", "chance_avg_main_hits", "p_value_vs_chance",
                     "¿Le gana al azar? (p<0.05)", "super_hit_rate", "chance_super_hit_rate"]]
            .style.format({
                "avg_main_hits": "{:.2f}", "chance_avg_main_hits": "{:.2f}", "p_value_vs_chance": "{:.3f}",
                "super_hit_rate": "{:.3f}", "chance_super_hit_rate": "{:.3f}",
            }),
            use_container_width=True,
        )
        st.caption(
            "Con pocas ventanas, incluso un modelo sin ninguna señal real puede parecer mejor o peor que el "
            "azar por pura varianza — usa el p-valor, no solo el promedio, y desconfía de una sola corrida con "
            "pocas ventanas."
        )
