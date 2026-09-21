"""The stake log, rendered the one way it is safe to render: verdict first.

One module shared by all three domain pages, for the same reason `ui.py` owns
the copy and `mobile.py` owns the phone layout — three pages each building their
own log could not be reviewed as a set, and the rule this surface exists to hold
is precisely the kind that erodes one page at a time.

**The rule.** A profit-and-loss figure at the top of a betting log is the single
most misleading number this project could put on a screen. It looks like
evidence, it moves every day, and at the sample sizes a person actually reaches
it is noise wearing a currency symbol. So this renders, in this order: the
**corrected verdict**, what the log **could** have detected with this many bets,
and only then the money — under a caption saying why it is down there.

`core/ledger.py` puts the same rule in its column order, so the two would have
to be broken separately for a total to end up on top.

**Recording goes through the domain's own refusals.** The form here builds a
row and hands it to `core/ledger.py`, which rejects a stake on an event that has
already happened exactly as it would from a script. The dashboard is not a way
round the guards; it is another caller of them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from core.ledger import LedgerError, load, open_bets, record, status, summary
from dashboard.ui import HELP, plain_verdict, section

# Spanish labels for the ledger's code-facing columns.
COLUMN_ES = {
    "event_date": "Fecha del evento", "label": "Estrategia", "selection": "Apuesta",
    "stake": "Monto", "price": "Cuota", "note": "Nota", "won": "Ganó",
    "payout": "Devuelto", "recorded_at": "Registrada", "settled_at": "Liquidada",
}

MONEY_COLUMNS = ("staked", "returned", "profit")


def render(path, title, selection_help, default_label="manual", price_help=None,
           selection_options=None):
    """The whole surface: record a stake, then read the log with the verdict on top.

    `path` is the domain's ledger file. `selection_options` turns the free-text
    selection into a picker where the page already knows the choices — a fixture
    and an outcome, a generated ticket — which is the only part of this that is
    domain-specific.
    """
    section(title, "log_tab")
    _render_form(path, selection_help, default_label, price_help, selection_options)
    st.divider()
    _render_verdict(path)
    _render_rows(path)


def _render_form(path, selection_help, default_label, price_help, selection_options):
    with st.form("registrar_apuesta", clear_on_submit=True):
        c1, c2 = st.columns(2)
        event_date = c1.date_input("Fecha del evento", help=HELP["log_event_date"])
        label = c2.text_input("Estrategia", value=default_label, help=HELP["log_label"])

        if selection_options:
            selection = st.selectbox("Apuesta", selection_options, help=selection_help)
        else:
            selection = st.text_input("Apuesta", help=selection_help)

        c3, c4 = st.columns(2)
        stake = c3.number_input("Monto", min_value=0.01, value=1.0, step=0.5,
                                help=HELP["log_stake"])
        price = c4.number_input("Cuota (decimal)", min_value=1.01, value=2.0, step=0.05,
                                help=price_help or HELP["log_price"])
        note = st.text_input("Nota", help=HELP["log_note"])

        if st.form_submit_button("Registrar"):
            try:
                record(path, event_date, label, selection, stake, price, note=note)
            except LedgerError as exc:
                # The guard speaking, not a validation message invented here: the
                # dashboard is another caller of the refusals, not a way round them.
                st.error(f"**No se registró.** {exc}")
            else:
                st.success(
                    f"Registrada: {selection} · {stake:g} a cuota {price:g} "
                    f"para el {event_date:%Y-%m-%d}."
                )


def _render_verdict(path):
    state = status(path)
    table = summary(path, by_label=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Apuestas registradas", state["n_rows"])
    c2.metric("Liquidadas", state["n_settled"], help=HELP["log_settled"])
    c3.metric("Abiertas", state["n_open"])

    if table.empty:
        st.info(
            "Todavía no hay ninguna apuesta liquidada, así que no hay nada que concluir. "
            "Eso no es un problema del registro: es lo que un registro joven puede decir."
        )
        return

    overall = summary(path).iloc[0]
    detectable = overall["min_detectable_return"]
    plain_verdict(
        bool(overall["beats_breakeven_corrected"]),
        ("Le gana al punto de equilibrio, con la corrección aplicada."
         if overall["beats_breakeven_corrected"]
         else "No hay evidencia de que le gane al punto de equilibrio."),
        (f"p (una cola) = {overall['p_value_greater']:.3f} · "
         f"umbral corregido = {overall['bonferroni_threshold']:.4f} · "
         f"intervalo del retorno = [{overall['ci_low']:+.1%}, {overall['ci_high']:+.1%}]"),
        good_is_pass=True,
    )
    st.caption(
        f"Con **{int(overall['n_settled'])} apuestas liquidadas**, lo más pequeño que este "
        f"registro podía haber detectado es un retorno de **{detectable:+.1%}** por unidad "
        "apostada. Un «no le gana» con pocas apuestas habla del tamaño de la muestra, no de la "
        "estrategia — y baja con la raíz de N, así que cuatro veces más apuestas lo reducen a la "
        "mitad.",
        help=HELP["log_mde"],
    )

    # The money, deliberately after the verdict and said to be secondary.
    st.markdown("**Dinero**", help=HELP["log_money"])
    money = summary(path).iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Apostado", f"{money['staked']:,.2f}")
    m2.metric("Devuelto", f"{money['returned']:,.2f}")
    m3.metric("Resultado", f"{money['profit']:+,.2f}", help=HELP["log_pnl"])
    st.caption(
        "Va aquí abajo a propósito. Un resultado acumulado arriba del todo es la cifra más "
        "engañosa que esta página podría mostrar: parece evidencia, cambia todos los días y con "
        "las cantidades de apuestas que alcanza una persona no distingue una ventaja real de una "
        "racha.",
        help=HELP["log_pnl"],
    )

    if len(table) > 1:
        st.markdown("**Por estrategia**", help=HELP["log_by_label"])
        st.dataframe(
            table.rename(columns={
                "label": "Estrategia", "n_settled": "Liquidadas",
                "beats_breakeven_corrected": "Le gana (corregido)",
                "p_value_greater": "p", "min_detectable_return": "Detectable",
                "roi": "Retorno", "profit": "Resultado"})
            [["Estrategia", "Liquidadas", "Le gana (corregido)", "p", "Detectable",
              "Retorno", "Resultado"]]
            .style.format({"p": "{:.3f}", "Detectable": "{:+.1%}", "Retorno": "{:+.1%}",
                           "Resultado": "{:+,.2f}"}),
            use_container_width=True, hide_index=True)
        st.caption(
            "El umbral ya está dividido entre el número de estrategias: leer la mejor de cinco "
            "es el mismo error que leer el mejor de cinco modelos.")


def _render_rows(path):
    ledger = load(path)
    if ledger.empty:
        return

    still_open = open_bets(ledger=ledger)
    if not still_open.empty:
        st.markdown("**Abiertas**", help=HELP["log_open"])
        st.dataframe(_readable(still_open, ("event_date", "label", "selection", "stake", "price")),
                     use_container_width=True, hide_index=True)

    settled = ledger[ledger["settled_at"].notna()]
    if not settled.empty:
        st.markdown("**Liquidadas**", help=HELP["log_settled"])
        st.dataframe(
            _readable(settled.sort_values("event_date", ascending=False),
                      ("event_date", "label", "selection", "stake", "price", "won", "payout")),
            use_container_width=True, hide_index=True)


def _readable(frame, columns):
    out = frame[list(columns)].copy()
    if "event_date" in out:
        out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    return out.rename(columns=COLUMN_ES)
