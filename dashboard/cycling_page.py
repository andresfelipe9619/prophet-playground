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

import numpy as np

from cycling.baseline import (
    baseline_frame,
    form_worths,
    predicted_order,
    uniform_worths,
    win_probabilities,
)
from cycling.common import DEFAULT_DATA_DIR, FINISHED, STAGE, format_seconds
from cycling.evaluation import compare_forecasters, race_groups
from cycling.plackett_luce import PlackettLuce
from cycling.processor import (
    ResultFormatError,
    check_result_format,
    load_races,
    preprocess_results,
    time_order_violations,
)
from cycling.sample_data import generate_stage_race
from cycling.scoring import METRICS, spearman, top_n_accuracy

from dashboard.ui import HELP, chart, glossary, plain_verdict, section

# Spanish labels for the code-facing constants in cycling/common.py.
# Short enough to fit a metric tile; the full sentence lives in the caption
# under it, where there is room to say what each kind's `rank` actually means.
KIND_ES = {"stage": "Etapa", "one_day": "Un día", "gc": "General"}
KIND_ES_LONG = {"stage": "resultado de etapa", "one_day": "clásica de un día",
                "gc": "clasificación general"}
STATUS_ES = {"FIN": "Llegó", "DNF": "Abandonó", "DNS": "No salió",
             "DSQ": "Descalificado", "OTL": "Fuera de control", "NR": "Sin clasificar"}

# Spanish labels for the code-facing forecaster and metric names.
FORECASTER_ES = {"ranking": "Ranking previo", "plackett_luce": "Modelo (Plackett-Luce)",
                 "uniform": "Sorteo uniforme (no es línea base)"}
METRIC_ES = {"plackett_luce": "Orden completo (Plackett-Luce) — el veredicto",
             "winner_log": "Solo el ganador (log)", "winner_brier": "Solo el ganador (Brier)"}


@st.cache_data(show_spinner=False)
def _fit_and_forecast(_results, cache_key, riders, as_of):
    """The ranking baseline and the fitted model for one race, from prior results only.

    `_results` is underscore-prefixed because the frame carries `.attrs` that
    Streamlit cannot hash; identity comes from `cache_key`. Everything here is
    built from results strictly before `as_of` — the whole point of the surface
    is that it could have been produced before the race.
    """
    history = _results[_results["ds"] < as_of]
    ranking = form_worths(history, riders, as_of=as_of)
    model = PlackettLuce.fit(history).worths_for(riders) if len(history) else None
    return ranking, model


@st.cache_data(show_spinner=False)
def _evaluate(_results, cache_key, metric, min_history):
    """Walk-forward comparison of the model and a uniform draw against the ranking."""
    return compare_forecasters(
        _results,
        {"ranking": lambda h, r, a: form_worths(h, r, as_of=a),
         "plackett_luce": lambda h, r, a: PlackettLuce.fit(h).worths_for(r),
         "uniform": lambda h, r, a: uniform_worths(len(r))},
        baseline="ranking", metric=metric, min_history=min_history)


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
        "El objetivo aquí es un **orden de llegada** de ~180 ciclistas, no un resultado de tres "
        "vías, y su línea base honesta es el mercado donde hay precio y si no el **ranking previo**. "
        "Un sorteo uniforme sobre la lista de salida no es una línea base: le da 0,55% a cada uno y "
        "cualquier pronóstico le gana sabiendo un solo nombre. Si una palabra no te suena, está en "
        "el **Glosario** de la barra lateral."
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
    glossary()

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

    tabs = st.tabs(["1 · Datos", "2 · Abandonos", "3 · Tiempos", "4 · Pronóstico",
                    "5 · ¿Le gana al ranking?"])

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

    # -------------------------------------------------------------- Pronóstico
    with tabs[3]:
        render_forecast_tab(results)

    # ------------------------------------------------------------- Evaluación
    with tabs[4]:
        render_evaluation_tab(results)


def render_forecast_tab(results):
    """The ranking baseline and the fitted model for one race, side by side.

    Never the model alone. A win probability of 12% means nothing without
    knowing that a uniform draw over 180 riders would say 0.55% and that the
    ranking — the thing the model actually has to beat — says something else
    again.
    """
    section("Quién debería ganar", "cy_tab_pronostico")

    groups = race_groups(results)
    if len(groups) < 2:
        st.info(
            "Hace falta más de una carrera en los datos: el pronóstico de una carrera se "
            "construye con las **anteriores**, así que la primera no tiene con qué."
        )
        return

    labels = [f"{key[0]} · {KIND_ES.get(key[1], key[1])}"
              + (f" · etapa {int(key[2])}" if key[2] == key[2] else "")
              for key, _ in groups]
    # The first race has nothing before it to forecast from, so the options start
    # at 1 — which makes the last option's position len(groups) - 2, not - 1.
    options = range(1, len(groups))
    picked = st.selectbox("Carrera", options, format_func=lambda i: labels[i],
                          index=len(options) - 1, help=HELP["cy_race_pick"])
    key, group = groups[picked]
    as_of = group["ds"].min()
    riders = list(group["rider"])

    st.caption(
        f"Pronóstico para **{labels[picked]}** ({as_of:%Y-%m-%d}), construido solo con las "
        f"{picked} carrera(s) anteriores. La carrera misma no se usa para nada de lo que sigue.",
        help=HELP["cy_baseline_pick"],
    )

    ranking, model = _fit_and_forecast(results, (len(results), str(as_of), len(riders)),
                                       riders, as_of)
    if model is None:
        st.info("No hay historia previa suficiente para ajustar el modelo.")
        return

    top_n = st.slider("Tamaño del «top N»", 3, 30, 10)
    table = baseline_frame(riders, model, n=top_n, n_samples=2000, seed=0)
    ranking_probabilities = dict(zip(riders, win_probabilities(ranking)))
    table["p_win_ranking"] = table["rider"].map(ranking_probabilities)

    c1, c2, c3 = st.columns(3)
    favourite = table.iloc[0]
    c1.metric("Favorito del modelo", favourite["rider"])
    c2.metric("Su probabilidad de ganar", f"{favourite['p_win']:.1%}", help=HELP["cy_win_probs"])
    c3.metric("Si fuera un sorteo uniforme", f"{1 / len(riders):.2%}",
              help=HELP["cy_uniform_warning"])
    st.caption(
        f"El favorito del **ranking previo** es **{predicted_order(ranking, riders)[0]}**. "
        "Que los dos coincidan o no es lo interesante de esta pantalla; cuál de los dos acierta "
        "más, sobre muchas carreras, lo dice **5 · ¿Le gana al ranking?**",
        help=HELP["cy_model_vs_baseline"],
    )

    head = table.head(top_n)
    figure = go.Figure()
    figure.add_bar(x=head["rider"], y=head["p_win"], name="Modelo")
    figure.add_bar(x=head["rider"], y=head["p_win_ranking"], name="Ranking previo")
    figure.add_hline(y=1 / len(riders), line_dash="dot", line_color="gray",
                     annotation_text="Sorteo uniforme")
    figure.update_layout(barmode="group", yaxis_title="Probabilidad de ganar",
                         yaxis_tickformat=".1%", height=380)
    chart(figure, f"Probabilidad de ganar — los {top_n} primeros del modelo", "cy_win_probs")

    st.markdown(f"**Probabilidad de entrar en el top {top_n}**", help=HELP["cy_top_n"])
    st.dataframe(
        head[["predicted_rank", "rider", "p_win", "p_win_ranking", f"p_top_{top_n}"]]
        .rename(columns={"predicted_rank": "#", "rider": "Ciclista", "p_win": "Gana (modelo)",
                         "p_win_ranking": "Gana (ranking)",
                         f"p_top_{top_n}": f"Top {top_n} (modelo)"})
        .style.format({"Gana (modelo)": "{:.2%}", "Gana (ranking)": "{:.2%}",
                       f"Top {top_n} (modelo)": "{:.1%}"}),
        use_container_width=True, hide_index=True)

    figure = go.Figure(go.Scatter(
        x=table["p_win_ranking"], y=table["p_win"], mode="markers", text=table["rider"]))
    limit = float(max(table["p_win"].max(), table["p_win_ranking"].max())) * 1.05
    figure.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit,
                     line=dict(dash="dash", color="gray"))
    figure.update_layout(xaxis_title="Ranking previo", yaxis_title="Modelo",
                         xaxis_tickformat=".1%", yaxis_tickformat=".1%", height=380)
    chart(figure, "Dónde discrepan el modelo y el ranking", "cy_model_vs_baseline")

    # Descriptive only, and said so: these look backwards at a race that has
    # already happened, which is the one thing the forecast above does not do.
    actual = group[(group["status"] == FINISHED) & group["rank"].notna()]
    if len(actual) > 3:
        order = list(table["rider"])
        d1, d2, d3 = st.columns(3)
        d1.metric("Correlación de rango (modelo)", f"{spearman(order, group):.2f}")
        d2.metric(f"Aciertos en el top {top_n}", f"{top_n_accuracy(order, group, n=top_n):.0%}")
        d3.metric("Ganó", actual.sort_values("rank")["rider"].iloc[0])
        st.caption(
            "Estas tres cifras miran **hacia atrás**, a una carrera que ya pasó, y son de una "
            "sola carrera: no son un veredicto ni de lejos. La correlación de rango además premia "
            "acertar el centro del pelotón, que es la parte fácil. El veredicto está en la "
            "pestaña siguiente."
        )


def render_evaluation_tab(results):
    """The walk-forward verdict against the ranking baseline."""
    section("¿Le gana el modelo al ranking previo?", "cy_tab_evaluacion")

    groups = race_groups(results)
    if len(groups) < 5:
        st.info(
            f"Solo hay {len(groups)} carrera(s) puntuable(s) en estos datos. Hacen falta varias "
            "para que la comparación signifique algo — carga una vuelta completa."
        )
        return

    c1, c2 = st.columns(2)
    metric = c1.selectbox("Regla de puntuación", list(METRICS),
                          format_func=lambda m: METRIC_ES[m], help=HELP["cy_metric"])
    min_history = c2.slider("Carreras de historia antes de empezar a puntuar", 1,
                            max(2, len(groups) // 2), min(4, max(2, len(groups) // 2)))

    if st.button("Correr la evaluación"):
        with st.spinner("Reajustando el modelo carrera por carrera…"):
            st.session_state["cy_eval"] = _evaluate(
                results, (len(results), metric, min_history), metric, min_history)

    stored = st.session_state.get("cy_eval")
    if stored is None:
        st.caption(
            "Cada carrera se puntúa dos veces — con el modelo y con el ranking previo — usando "
            "solo lo anterior a su fecha. La diferencia media entre las dos es el resultado."
        )
        return

    table, scores = stored
    model_row = table[table["forecaster"] == "plackett_luce"]
    if not model_row.empty:
        row = model_row.iloc[0]
        plain_verdict(
            bool(row["beats_baseline_corrected"]),
            "El modelo le gana al ranking previo"
            if row["beats_baseline_corrected"] else
            "El modelo no le gana al ranking previo, no con estos datos",
            f"Puntuó {row['model_score']:.4f} contra {row['baseline_score']:.4f} del ranking, "
            f"sobre {int(row['n_races'])} carreras (ventaja {row['effect']:+.4f}, IC 95% "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]). "
            "Con veintitantas carreras el intervalo es ancho: que cruce el cero significa que "
            "esta muestra no tuvo resolución para decidirlo, no que la ventaja sea cero.",
        )

    display = table.copy()
    display["Pronosticador"] = display["forecaster"].map(FORECASTER_ES)
    display["Ventaja (IC 95%)"] = display.apply(
        lambda r: f"{r['effect']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", axis=1)
    display["¿Le gana? (corregido)"] = display["beats_baseline_corrected"].map(
        {True: "Sí", False: "No"})
    st.markdown("**Veredicto por pronosticador**", help=HELP["cy_eval_table"])
    st.dataframe(
        display[["Pronosticador", "n_races", "model_score", "baseline_score", "skill_score",
                 "Ventaja (IC 95%)", "p_value_greater", "bonferroni_threshold",
                 "¿Le gana? (corregido)"]]
        .rename(columns={"n_races": "Carreras", "model_score": "Puntuación",
                         "baseline_score": "Ranking", "skill_score": "Skill",
                         "p_value_greater": "p (una cola)",
                         "bonferroni_threshold": "Umbral corregido"})
        .style.format({"Puntuación": "{:.4f}", "Ranking": "{:.4f}", "Skill": "{:+.4f}",
                       "p (una cola)": "{:.4f}", "Umbral corregido": "{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "Menor puntuación es mejor. El **sorteo uniforme** está en la tabla a propósito y debe "
        "salir peor que el ranking: es la demostración de que no es una línea base, sino una "
        "forma de hacer ver brillante a cualquier modelo.",
        help=HELP["cy_uniform_warning"],
    )

    figure = go.Figure()
    for name, group in scores.groupby("forecaster"):
        group = group.sort_values("ds")
        figure.add_scatter(x=group["ds"], y=group["score"], mode="lines+markers",
                           name=FORECASTER_ES.get(name, name))
    figure.update_layout(yaxis_title=f"{METRIC_ES[metric]} (menor es mejor)", height=380)
    chart(figure, "Puntuación carrera por carrera", "cy_eval_chart")

    st.info(
        f"**Resolución de esta corrida.** Son {int(table['n_races'].max())} carreras puntuadas. "
        "Una gran vuelta son 21 y una temporada de clásicas unas pocas decenas, así que aquí un "
        "«no le gana» habla del tamaño de la muestra todavía más que en los otros dominios. "
        "Y en las etapas de sprint el orden de llegada es casi ruido: ningún pronóstico puede "
        "ganarle al ranking ahí, ni debería.",
        icon="ℹ️",
    )
