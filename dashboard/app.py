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
from analysis.tickets import (
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
    ticket_from_predictions,
)
from models.baseline import expected_main_matches, most_frequent_pick
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
from utils.processor import (
    check_draw_format,
    current_format_mask,
    load_and_preprocess,
    preprocess_draws,
)
from utils.sample_data import load_sample_and_preprocess

st.set_page_config(page_title="Baloto Analytics", layout="wide")

MIN_TRAIN_FLOOR = 20  # below this the models have nothing to learn from

# Every explanatory tooltip in the UI, in one place.
#
# Streamlit renders `help=` as a small ⓘ next to the element and shows the text
# on hover, which is the right home for "what am I looking at?" — it is there
# when you want it and out of the way when you don't. Keeping the copy in one
# dict rather than inline at each call site is what makes it reviewable as a
# whole: this project's rule is that no chart or table appears without saying
# what it does *not* mean, and that is only checkable if the texts sit together.
#
# Spanish, like the rest of the UI. Docstrings and comments stay English.
HELP = {
    # -- Resumen
    "tab_resumen": "Estado general de los datos cargados y el veredicto sobre si tus sorteos se comportan "
                   "como un sorteo justo. Empieza por aquí: si los datos vienen mal, todo lo demás sobra.",
    "n_draws": "Cuántos sorteos hay cargados después de filtrar. Más sorteos = pruebas estadísticas más "
               "sensibles. Por debajo de ~200 casi nada es concluyente.",
    "date_from": "Fecha del sorteo más antiguo del archivo cargado.",
    "date_to": "Fecha del sorteo más reciente. Si está muy atrás, actualiza con "
               "`python -m utils.scraper --years <año>`.",
    "sorted_flag": "Muchas fuentes publican las 5 balotas ordenadas de menor a mayor. Si es así, cada columna "
                   "deja de ser una balota al azar y pasa a ser un estadístico de orden (el mínimo, el 2do "
                   "menor...), lo que hace que las pruebas por posición marquen patrones falsos. La prueba "
                   "agrupada de abajo es inmune a esto.",
    "pooled_test": "Prueba chi-cuadrado agrupada: junta todas las posiciones y solo pregunta si cada número "
                   "sale aproximadamente la misma cantidad de veces. Al no mirar en qué columna cayó cada "
                   "balota, no la engaña que los datos vengan ordenados.",
    "pooled_main_p": "p-valor de la prueba agrupada sobre las 5 balotas principales. Alto (>0.05) = sin "
                     "evidencia contra la uniformidad, que es el resultado sano. Bajo sería raro de verdad, "
                     "y más probablemente indica un problema de datos que una lotería vencible.",
    "pooled_super_p": "Lo mismo para la superbalota, evaluada aparte porque su rango es 1-16 y no 1-43.",

    # -- Probabilidades
    "tab_probabilidades": "La única pestaña con respuestas exactas y sin modelos: son combinatoria pura, no "
                          "dependen de tu histórico ni de ninguna predicción.",
    "jackpot_odds": "Probabilidad de acertar las 5 principales más la superbalota, calculada exactamente. "
                    "No cambia con la estrategia, la suerte ni la fecha.",
    "total_combos": "Todas las jugadas posibles: C(43,5) × 16. Todas son igual de probables.",
    "prize_table": "Los montos son de ejemplo y debes reemplazarlos por la tabla oficial vigente: varias "
                   "categorías son variables y el premio mayor se acumula. Las probabilidades de la izquierda "
                   "son exactas y no cambian con lo que escribas.",
    "ticket_price": "Precio de un tiquete. Se usa para calcular el valor esperado y el RTP de abajo.",
    "ev_section": "Cuánto vale jugar, dados los premios que escribiste arriba. Es aritmética exacta sobre las "
                  "probabilidades reales, no una simulación ni una estimación.",
    "expected_return": "Promedio que devuelve un tiquete a largo plazo, antes de restar lo que costó.",
    "expected_value": "Retorno esperado menos el precio del tiquete. Negativo significa que cada jugada pierde "
                      "esa cantidad en promedio. Ninguna forma de elegir números lo cambia.",
    "rtp": "Qué porcentaje de lo apostado devuelve el juego a largo plazo. Una máquina tragamonedas ronda el "
           "90%; una lotería suele estar muy por debajo.",
    "any_prize": "Probabilidad de llevarte *algo*, aunque sea la categoría más baja que pague.",
    "breakeven": "Cuánto tendría que acumularse el premio mayor para que el valor esperado llegue a cero. Ojo: "
                 "aun superándolo, el acumulado se reparte entre todos los ganadores y hay retención.",
    "category_chart": "Probabilidad exacta de cada categoría, en escala logarítmica porque abarcan varios "
                      "órdenes de magnitud. Las barras más bajas son las que pagan más.",

    # -- Frecuencia y gaps
    "tab_frecuencia": "Cuántas veces ha salido cada número y cuánto lleva sin salir. Vista descriptiva: "
                      "describe el pasado, no anticipa el futuro.",
    "freq_chart": "Veces que salió cada número en esta posición, contra la línea punteada de lo que se "
                  "esperaría si todo fuera uniforme. Ojo con el caso ordenado: si el Resumen dice que tus "
                  "balotas vienen guardadas de menor a mayor, esta columna no es una balota al azar sino un "
                  "estadístico de orden, y verás una escalera marcada (la 'Balota 1' es siempre el mínimo, "
                  "así que los números bajos dominan). Eso es el orden, no un patrón del sorteo. Sin ese "
                  "efecto, las diferencias que quedan son ruido de muestreo normal: con unos cientos de "
                  "sorteos y 43 números ninguna barra cae exactamente en la línea. El veredicto real está "
                  "en la prueba agrupada del Resumen.",
    "gaps_section": "Cada cuánto suele aparecer un número y cuánto lleva ausente. El *overdue score* es el "
                    "heurístico de 'ya se demoró, le toca'. Para sorteos independientes no tiene poder "
                    "predictivo: es la falacia del jugador. Está aquí porque mucha gente lo busca, no porque "
                    "sirva.",

    # -- Hot / cold
    "tab_hotcold": "Qué números vienen saliendo más (o menos) que su promedio histórico en la ventana "
                   "reciente que elijas.",
    "hotcold_chart": "Diferencia entre el % de apariciones en los últimos sorteos y el % de todo el "
                     "histórico. Rojo = por encima, azul = por debajo. Con ventanas cortas hay muy pocas "
                     "observaciones por número, así que estas barras se mueven mucho aunque el sorteo no "
                     "haya cambiado en nada.",
    "hotcold_window": "Cuántos sorteos recientes componen la ventana 'caliente'. Más corta = más ruido.",

    # -- Aleatoriedad
    "tab_aleatoriedad": "Las pruebas formales: ¿hay alguna estructura explotable en estos sorteos? Lo "
                        "esperable, y lo sano, es que la respuesta sea no.",
    "verdict_table": "Tres pruebas por posición. chi-cuadrado: ¿salen todos los números con la misma "
                     "frecuencia? runs test: ¿hay rachas por encima o por debajo de la mediana? Ljung-Box: "
                     "¿un sorteo dice algo del siguiente? Recuerda que al correr 6 posiciones a la vez, ~1 de "
                     "cada 20 pruebas marca 'No' por puro azar.",
    "acf_section": "Autocorrelación: cuánto se parece la serie a sí misma desplazada N sorteos. Es la prueba "
                   "directa de si un modelo de series de tiempo (ARIMA, Prophet) tiene algo que aprender aquí.",
    "acf_chart": "Cada barra es la correlación con el sorteo N posiciones atrás. Las líneas punteadas son la "
                 "banda de significancia: barras dentro de la banda son indistinguibles de cero. Un ACF "
                 "totalmente dentro de la banda es exactamente lo que produce un proceso sin memoria.",
    "ljung_box": "Prueba conjunta sobre todos los lags a la vez. p alto = no hay autocorrelación detectable, "
                 "es decir, no hay 'memoria' que un modelo pueda explotar.",

    # -- Forecast
    "tab_forecast": "Corre un modelo sobre tu histórico y pídele una sugerencia para el próximo sorteo. Es un "
                    "ejercicio de forecasting, no una predicción confiable.",
    "model_choice": "FrequencyBaseline juega el número más frecuente de cada posición (la referencia a "
                    "vencer). AutoARIMA/AutoETS/AutoTheta son modelos clásicos de series de tiempo. Prophet "
                    "es el más lento. XGBoost usa lags y medias móviles. Ninguno supera al azar en el "
                    "backtest — para eso está esa pestaña.",

    # -- Jugadas
    "tab_jugadas": "Genera jugadas, verifícalas contra sorteos reales y mide si tu forma de elegirlas le gana "
                   "al azar. Generar números es válido; lo que ninguna estrategia logra es hacer una jugada "
                   "más probable que otra.",
    "n_tickets": "Cuántas jugadas generar de una vez.",
    "distinct_numbers": "Cuántos números distintos cubre el conjunto de jugadas entre todas.",
    "pool_coverage": "Qué porcentaje de los 43 números toca tu portafolio. Cubrir más reparte los resultados "
                     "sobre el conjunto; no mejora ninguna jugada individual.",
    "portfolio_odds": "Probabilidad del premio mayor con todas estas jugadas juntas. Comprar N jugadas divide "
                      "la probabilidad entre N — es aritmética, y cuesta N veces más.",
    "check_input": "Escribe una jugada real (la tuya, por ejemplo) y mira cómo le habría ido en cada sorteo "
                   "de tu histórico.",
    "history_chart": "En cuántos sorteos de tu histórico esa jugada habría caído en cada categoría. Compáralo "
                     "con las probabilidades exactas de la pestaña Probabilidades: cualquier otra jugada da "
                     "una distribución estadísticamente equivalente.",
    "experiment_intro": "El experimento honesto: para cada sorteo se generan jugadas usando solo los sorteos "
                        "anteriores, y se comparan los aciertos contra la expectativa exacta del azar.",
    "draws_back": "Cuántos sorteos históricos evaluar. Más sorteos = más poder estadístico.",
    "per_draw": "Cuántas jugadas generar para cada sorteo evaluado.",
    "strategy_chart": "Aciertos promedio por jugada de cada estrategia, contra lo que da el azar puro "
                      "(0.58 de 5). Barras casi iguales es el resultado esperado y correcto.",
    "strategy_table": "La columna de veredicto usa umbral corregido por comparaciones múltiples (Bonferroni): "
                      "al probar varias estrategias a la vez, alguna parece ganadora por azar mucho más "
                      "seguido de lo que sugiere un 0.05 suelto.",
    "stability_section": "Una sola corrida es un sorteo de un proceso ruidoso: con α = 0.05, una estrategia "
                         "sin ninguna ventaja parece ganadora ~1 de cada 20 veces. Esto repite el "
                         "experimento con varias semillas y cuenta cuántas veces marcó ganador.",
    "stability_table": "`random` no puede tener ventaja: su tasa de marcado es tu piso de falsos positivos "
                       "medido. Una estrategia que no marque claramente más seguido que ella no ha "
                       "demostrado nada.",

    # -- Backtest
    "tab_backtest": "El veredicto del proyecto: cada modelo se entrena solo con datos anteriores al sorteo "
                    "que intenta predecir, y se compara contra lo que realmente salió. Lo que importa no es "
                    "cuántos aciertos, sino cuántos más que el azar puro.",
    "experiment_choice": "Últimos N sorteos da un promedio sobre la cola del histórico. Corte por fecha "
                         "responde la pregunta concreta: entreno con todo hasta julio, ¿qué habría predicho "
                         "para agosto y septiembre, que ya sabemos cómo salieron?",
    "n_windows": "Cuántos sorteos recientes dejar fuera del entrenamiento y evaluar. Con menos de ~30, una "
                 "sola corrida es una anécdota.",
    "min_train": "Cuántos sorteos como mínimo debe tener el modelo para entrenar antes de la primera ventana.",
    "include_prophet": "Prophet reajusta un modelo por posición y por ventana, así que multiplica el tiempo "
                       "de corrida. Déjalo apagado salvo que lo necesites.",
    "cutoff_date": "El modelo se entrena con todos los sorteos hasta esta fecha (inclusive) y predice los "
                   "posteriores, que ya sabemos cómo salieron.",
    "holdout_mode": "Reentrenar en cada sorteo es lo que harías jugando de verdad: antes de cada sorteo "
                    "reajustas el modelo con todo lo conocido hasta ahí. Entrenar una vez es la prueba "
                    "literal 'ajusto en julio y proyecto agosto y septiembre a ciegas' — más dura, porque el "
                    "modelo no ve nada nuevo.",
    "holdout_detail": "Sorteo por sorteo: los números que salieron de verdad y cuántos acertó cada modelo. "
                      "Una fila con 3 aciertos no es señal: acertar 3 o más de 5 en 43 pasa cerca del 1% de "
                      "las veces por azar, así que con varios modelos y una docena de sorteos es esperable "
                      "ver alguna. Lo que decide es el promedio.",
    "holdout_chart": "Aciertos de cada modelo en cada sorteo del periodo de prueba. La línea punteada es lo "
                     "que da el azar (0.58 de 5). Los picos por encima y por debajo son varianza normal.",
    "summary_chart": "Aciertos promedio de cada modelo contra el promedio del azar puro. Si las barras se ven "
                     "casi iguales, ese es el resultado esperado y correcto para una lotería justa.",
    "summary_table": "El p-valor es de una cola: mide si el modelo es *mejor* que el azar, no solo distinto. "
                     "Lee la columna corregida: al probar varios modelos contra los mismos sorteos, alguno "
                     "pasa el 5% por suerte mucho más seguido de lo que ese 5% sugiere. La corrección de "
                     "Bonferroni baja el umbral a 0.05 dividido entre el número de modelos.",
}


def section(title, help_key):
    """A subheader with the ⓘ that explains the section it opens."""
    st.subheader(title, help=HELP[help_key])


def chart(fig, title, help_key):
    """Render a Plotly figure under a titled line carrying its own ⓘ.

    The title moves out of the figure and into Streamlit so every chart in the
    dashboard gets the same typography and the same explain-on-hover affordance;
    Plotly's own title has nowhere to hang a help icon.
    """
    st.markdown(f"**{title}**", help=HELP[help_key])
    # `title=None` leaves Plotly rendering the string "undefined"; an empty text
    # is what actually clears it.
    fig.update_layout(title={"text": ""}, margin=dict(t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)


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
    current_format_only = st.checkbox(
        "Solo sorteos del formato actual", value=True,
        help="Baloto cambió de reglas en abril de 2017 (antes: 6 balotas del 1 al 45, sin "
             "superbalota). Los dos formatos se publican igual, así que un histórico largo suele "
             "mezclarlos. Desmarca solo si sabes lo que estás haciendo.",
    )

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

tabs = st.tabs([
    "Resumen", "Probabilidades y Valor Esperado", "Frecuencia y Gaps", "Hot / Cold",
    "Aleatoriedad", "Forecast", "Jugadas", "Backtest vs. Azar",
])

# ---------------------------------------------------------------- Resumen
with tabs[0]:
    section("Estado de los datos", "tab_resumen")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sorteos", n_draws, help=HELP["n_draws"])
    col2.metric("Desde", df["ds"].min().strftime("%Y-%m-%d"), help=HELP["date_from"])
    col3.metric("Hasta", df["ds"].max().strftime("%Y-%m-%d"), help=HELP["date_to"])
    pooled_main, pooled_super, sorted_flag = cached_pooled_tests(balls_expanded, n_columns)
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
        st.metric("p-valor (balotas principales, agrupadas)", f"{pooled_main['p_value']:.3f}",
                  help=HELP["pooled_main_p"])
        verdict_badge(pooled_main["p_value"] > 0.05)
    with c2:
        st.metric("p-valor (superbalota)", f"{pooled_super['p_value']:.3f}",
                  help=HELP["pooled_super_p"])
        verdict_badge(pooled_super["p_value"] > 0.05)
    st.caption(
        "p-valor alto (>0.05) = no hay evidencia contra la hipótesis de uniformidad, que es justamente lo "
        "esperable en un sorteo justo. Un p-valor bajo aquí sí sería una señal real y rara — vale la pena "
        "revisar la fuente de datos si eso pasa."
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

# ------------------------------------------------------------------ Forecast
with tabs[5]:
    section("Sugerencia para el próximo sorteo", "tab_forecast")
    st.warning(
        "Estos son ejercicios de forecasting, no predicciones confiables: para un sorteo justo, ningún modelo "
        "puede superar de forma sostenida la probabilidad teórica. Revisa la pestaña Backtest antes de confiar "
        "en cualquiera de estos números."
    )
    model_choice = st.selectbox("Modelo", ["FrequencyBaseline", "Prophet", *MODEL_NAMES, "XGBoost"],
                                 help=HELP["model_choice"])

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

    gen_tab, check_tab, exp_tab = st.tabs(["Generar", "Verificar", "Medir estrategias"])

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
                {"#": i + 1, "Balotas": " - ".join(str(n) for n in sorted(t.main)), "Superbalota": t.super_ball}
                for i, t in enumerate(tickets)
            ]), use_container_width=True, hide_index=True)

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
            st.markdown("**Resultado por estrategia**", help=HELP["strategy_table"])
            st.dataframe(
                display[["strategy", "n_tickets_evaluated", "avg_main_matches", "chance_avg_main_matches",
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
        include_prophet = c3.checkbox("Incluir Prophet (más lento)", value=False,
                                       help=HELP["include_prophet"])

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
                                          min_train=min_train, include_prophet=include_prophet)
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
        include_prophet = c3.checkbox("Incluir Prophet (más lento)", value=False, key="holdout_prophet",
                                       help=HELP["include_prophet"])

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
                                                    mode=mode, include_prophet=include_prophet)
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
        fig = go.Figure()
        fig.add_bar(x=summary["model"], y=summary["avg_main_hits"], name="Modelo")
        fig.add_bar(x=summary["model"], y=summary["chance_avg_main_hits"], name="Azar (esperado)")
        fig.update_layout(barmode="group", yaxis_title="Aciertos promedio")
        chart(fig, "Aciertos promedio (balotas principales) vs. azar", "summary_chart")

        display = summary.copy()
        display["¿Le gana al azar? (p<0.05)"] = display["beats_chance"].map({True: "Sí", False: "No"})
        display["¿Le gana? (corregido)"] = display["beats_chance_corrected"].map({True: "Sí", False: "No"})
        st.markdown("**Veredicto por modelo**", help=HELP["summary_table"])
        st.dataframe(
            display[["model", "n_windows", "avg_main_hits", "chance_avg_main_hits",
                     "p_value_better_than_chance", "¿Le gana al azar? (p<0.05)",
                     "bonferroni_threshold", "¿Le gana? (corregido)", "super_hit_rate",
                     "chance_super_hit_rate"]]
            .style.format({
                "avg_main_hits": "{:.2f}", "chance_avg_main_hits": "{:.2f}",
                "p_value_better_than_chance": "{:.3f}", "bonferroni_threshold": "{:.4f}",
                "super_hit_rate": "{:.3f}", "chance_super_hit_rate": "{:.3f}",
            }),
            use_container_width=True,
        )
        st.caption(
            "El p-valor es de una cola: mide si el modelo es *mejor* que el azar, no solo distinto "
            "(un modelo peor que el azar no cuenta como que le gana). **Lee la columna corregida**: "
            f"se prueban {len(summary)} modelos contra los mismos sorteos, así que hay "
            f"{len(summary)} oportunidades de que alguno pase el 5% por suerte — la corrección de "
            "Bonferroni baja el umbral a 0.05 dividido entre el número de modelos. Con pocas "
            "ventanas, incluso un modelo sin señal real puede parecer mejor o peor por pura varianza."
        )
