"""The cycling page: the result contract made visible, and nothing more.

Run through `dashboard/app.py`, which owns the sidebar's domain selector.

**What this page deliberately does not do.** There is no cycling model here yet,
no scoring rule for a finishing order, and not even the ranking baseline a model
would have to beat. So nothing on this page predicts or scores anything; it
describes the data.

What it *is* for is the part that costs most when it goes unnoticed. The three
things the contract protects — one kind of result per frame, non-finishers kept,
times stored as totals rather than gaps — are all invisible in the shape of a
frame, so this page puts them on screen: which kind of result is loaded, how many
riders abandoned, and whether any rider is timed faster than someone placed ahead
of them (the fingerprint of gaps stored as totals).
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cycling.common import DEFAULT_DATA_DIR, FINISHED, STAGE, format_seconds
from cycling.processor import (
    ResultFormatError,
    check_result_format,
    load_races,
    preprocess_results,
    time_order_violations,
)
from cycling.sample_data import generate_stage_race

from dashboard.ui import HELP, chart, section

# Spanish labels for the code-facing constants in cycling/common.py.
# Short enough to fit a metric tile; the full sentence lives in the caption
# under it, where there is room to say what each kind's `rank` actually means.
KIND_ES = {"stage": "Etapa", "one_day": "Un día", "gc": "General"}
KIND_ES_LONG = {"stage": "resultado de etapa", "one_day": "clásica de un día",
                "gc": "clasificación general"}
STATUS_ES = {"FIN": "Llegó", "DNF": "Abandonó", "DNS": "No salió",
             "DSQ": "Descalificado", "OTL": "Fuera de control", "NR": "Sin clasificar"}


def result_files(directory):
    """The result CSVs sitting in `directory`."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".csv"))


@st.cache_data(show_spinner=False)
def load_results(uploaded_bytes, paths):
    """Load results from an upload, from files, or fall back to a synthetic race.

    `load_races` is allowed to raise through: refusing to concatenate a stage
    result with a general classification is the guard this domain exists around,
    and catching it here to show *something* would defeat it.
    """
    if uploaded_bytes is not None:
        results = preprocess_results(pd.read_csv(io.BytesIO(uploaded_bytes)), validate=False)
        is_demo = False
    elif paths:
        results = load_races(list(paths), validate=False)
        is_demo = False
    else:
        results = preprocess_results(generate_stage_race(seed=0), validate=False)
        is_demo = True

    return results, is_demo, check_result_format(results)


def render():
    st.title("🚴 Ciclismo")
    st.caption(
        "Capa de datos. El objetivo aquí es un **orden de llegada** de ~180 ciclistas, no un resultado "
        "de tres vías, y su línea base honesta es el mercado donde hay precio y si no el ranking previo "
        "(puntos UCI/PCS). Ni la línea base ni los modelos existen todavía: esta página describe los "
        "datos y hace visible lo que el contrato protege."
    )

    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader("CSV de resultados", type="csv")
        directory = st.text_input("Carpeta de resultados", value=DEFAULT_DATA_DIR)
        available = result_files(directory)
        chosen = st.multiselect("Archivos", available, default=available[:1],
                                help=HELP["cy_files"])
        show_finishers_only = st.checkbox(
            "Ocultar los que no clasificaron", value=False, help=HELP["cy_finishers_only"])

    paths = tuple(os.path.join(directory, name) for name in chosen) if not uploaded else ()

    try:
        results, is_demo, report = load_results(
            uploaded.getvalue() if uploaded else None, paths)
    except ResultFormatError as exc:
        st.error(
            "**No se pudieron cargar estos archivos juntos.** Un archivo contiene un solo tipo de "
            "resultado: un puesto en una etapa y un puesto en la general son cantidades distintas — "
            "un sprint contra tres semanas de tiempo acumulado — y juntarlas dejaría una columna "
            "`rank` que significa dos cosas. Carga un tipo a la vez."
        )
        # The original message names the kinds involved; it is English because it
        # comes from the domain layer, where everything is.
        st.caption(f"Detalle: {exc}")
        st.stop()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if is_demo:
        st.info(
            "No hay archivos en la carpeta indicada ni se subió uno — mostrando una **carrera "
            "sintética** (tres semanas generadas con una fuerza latente por ciclista, llegadas en "
            "pelotón y abandonos). Sirve para recorrer la página, no para concluir nada. Descarga "
            "resultados reales con `python -m cycling.scraper --race tour-de-france --year 2024 "
            "--stages 1-21`."
        )

    kind = results.attrs.get("result_kind")
    violations = time_order_violations(results)
    non_finishers = results[results["status"] != FINISHED]

    if violations.any():
        st.error(
            f"**{int(violations.sum())} ciclistas están cronometrados más rápido que alguien que "
            "quedó por delante.** `time_seconds` debe ser tiempo total acumulado; esto es lo que "
            "parece una columna de diferencias al ganador. No uses los tiempos hasta revisar el "
            "scrapeo."
        )
    if len(non_finishers) == 0 and len(results):
        st.warning(
            "**No hay ni un abandono en todo el archivo.** Una carrera real los tiene, así que esto "
            "probablemente ya viene filtrado — y eso hace optimista cualquier medida de acierto: los "
            "ciclistas más difíciles de predecir son justo los que faltan."
        )

    # The filter is applied after the checks above on purpose, so the warning
    # about a frame with no abandons fires on the file as published rather than
    # on the filtered view.
    shown = results[results["status"] == FINISHED] if show_finishers_only else results

    tabs = st.tabs(["Datos", "Abandonos", "Tiempos"])

    # ------------------------------------------------------------------ Datos
    with tabs[0]:
        section("Estado de los datos", "cy_tab_datos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", len(results), help=HELP["cy_n_rows"])
        c2.metric("Tipo de resultado", KIND_ES.get(kind, kind or "—"), help=HELP["cy_kind"])
        c3.metric("Carreras", results["race"].nunique())
        c4.metric("Ciclistas", results["rider"].nunique())

        c1, c2, c3 = st.columns(3)
        c1.metric("No clasificados", len(non_finishers), help=HELP["cy_non_finishers"])
        c2.metric("Sin tiempo", int(results["time_seconds"].isna().sum()),
                  help=HELP["cy_missing_times"])
        c3.metric("Tiempos incoherentes", int(violations.sum()), help=HELP["cy_violations"])

        st.caption(
            f"Un archivo contiene **un solo tipo de resultado**: aquí, "
            f"{KIND_ES_LONG.get(kind, kind)}. "
            "Un puesto 4 en una etapa y un puesto 4 en la general son cantidades distintas — un "
            "sprint contra tres semanas de tiempo acumulado — así que `load_races` se niega a "
            "juntarlos en una misma columna `rank`."
        )

        if report:
            # The report is written in English, like everything in the domain
            # layer, and the banners above already say the part that matters in
            # Spanish. It is kept, folded away, because it carries the counts.
            with st.expander("Detalle técnico (en inglés)"):
                st.write(report["message"])

        st.markdown("**Primeras filas**", help=HELP["cy_table"])
        table = shown.head(30).copy()
        table["tiempo"] = table["time_seconds"].map(
            lambda s: "" if pd.isna(s) else format_seconds(s))
        table["estado"] = table["status"].map(lambda s: STATUS_ES.get(s, s))
        st.dataframe(table.drop(columns=["time_seconds", "status"]), use_container_width=True)

    # --------------------------------------------------------------- Abandonos
    with tabs[1]:
        section("Quién no llega al final", "cy_tab_abandonos")
        if not len(non_finishers):
            st.info("No hay no clasificados en estos datos.")
        else:
            counts = non_finishers["status"].value_counts()
            columns = st.columns(max(len(counts), 1))
            for column, (status, count) in zip(columns, counts.items()):
                column.metric(STATUS_ES.get(status, status), int(count),
                              help=HELP["cy_status_counts"])

        if kind == STAGE and results["stage"].notna().any():
            per_stage = results.groupby("stage").agg(
                clasificados=("status", lambda s: int((s == FINISHED).sum())),
                no_clasificados=("status", lambda s: int((s != FINISHED).sum())),
            ).reset_index()

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=per_stage["stage"], y=per_stage["clasificados"],
                                        mode="lines+markers", name="Clasificados"))
            figure.update_layout(xaxis_title="Etapa", yaxis_title="Ciclistas", height=340)
            chart(figure, "Pelotón restante por etapa", "cy_attrition")

            figure = go.Figure(go.Bar(x=per_stage["stage"], y=per_stage["no_clasificados"]))
            figure.update_layout(xaxis_title="Etapa", yaxis_title="Abandonos", height=300)
            chart(figure, "Abandonos por etapa", "cy_abandons_per_stage")

        st.caption(
            "Los abandonos se quedan en los datos, con `rank` vacío y un estado que dice por qué. "
            "No son aleatorios: se concentran en los ciclistas de peor forma, o sea en los que un "
            "modelo tenía menos claros. Filtrarlos convierte «predecir el orden de llegada» en "
            "«predecir el orden entre los que llegaron», que es un problema más fácil y que nadie "
            "puede apostar."
        )

    # ----------------------------------------------------------------- Tiempos
    with tabs[2]:
        section("Diferencias al primero", "cy_tab_tiempos")
        timed = results[results["rank"].notna() & results["time_seconds"].notna()]
        if not len(timed):
            st.warning("Estos resultados no traen tiempos utilizables.")
        else:
            groups = sorted(timed["stage"].dropna().unique()) if timed["stage"].notna().any() else []
            if groups:
                stage = st.selectbox("Etapa", groups, index=len(groups) - 1, help=HELP["cy_stage_pick"])
                group = timed[timed["stage"] == stage]
            else:
                group = timed

            gaps = group.sort_values("rank")
            leader = gaps["time_seconds"].iloc[0]
            figure = go.Figure(go.Scatter(
                x=gaps["rank"], y=gaps["time_seconds"] - leader, mode="lines+markers",
                text=gaps["rider"]))
            figure.update_layout(xaxis_title="Puesto", yaxis_title="Segundos respecto al primero",
                                 height=380)
            chart(figure, "Diferencia acumulada por puesto", "cy_gaps")

            c1, c2, c3 = st.columns(3)
            c1.metric("Tiempo del primero", format_seconds(leader), help=HELP["cy_leader_time"])
            c2.metric("Diferencia del último clasificado",
                      format_seconds(gaps["time_seconds"].iloc[-1] - leader))
            c3.metric("Con el mismo tiempo que el primero",
                      int((gaps["time_seconds"] == leader).sum()), help=HELP["cy_same_time"])

            st.caption(
                "La columna guarda **tiempo total**, no la diferencia que publica la página: el "
                "scraper suma cada diferencia al tiempo del ganador y, si falta el tiempo del "
                "ganador, deja todos los tiempos vacíos en lugar de escribir diferencias en una "
                "columna que significa totales. Un grupo entero con el mismo tiempo que el primero "
                "es una llegada en pelotón, no un error."
            )
