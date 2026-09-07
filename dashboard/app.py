"""The dashboard shell: one app, three domains, chosen from the sidebar.

Run with: streamlit run dashboard/app.py

This file owns only the things that are true of every domain — the page config,
the domain selector, and the dispatch. Each page lives in its own module and is
imported lazily, because Baloto's page alone pulls in statsforecast and xgboost
and there is no reason to pay for that while looking at cycling results.

**The three domains are not equally built, and the shell says so** rather than
presenting them as peers. Baloto has models, a chance baseline and a backtest;
football has its data contract, the market baseline, a Dixon-Coles model, a
proper scoring rule and a walk-forward backtest (only an Elo baseline is still
missing); cycling has its data contract and nothing above it. A launcher
that listed them identically would imply three finished products, which is
exactly the kind of quiet overclaim the rest of this project is built to avoid.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="Sports Analytics", layout="wide")

# Label -> (module, what exists there today). The second half is shown in the
# sidebar under the selector, so the state of each domain is visible before you
# click rather than discovered after.
DOMAINS = {
    "🎯 Baloto": (
        "dashboard.baloto_page",
        "Modelos, línea base de azar, backtest, jugadas y registro.",
    ),
    "⚽ Fútbol": (
        "dashboard.football_page",
        "Datos, línea base del mercado, modelo Dixon-Coles y backtest. Falta el Elo.",
    ),
    "🚴 Ciclismo": (
        "dashboard.cycling_page",
        "Solo datos. Sin línea base ni modelos todavía.",
    ),
}


def main():
    with st.sidebar:
        st.title("Deportes")
        choice = st.radio(
            "Dominio", list(DOMAINS), label_visibility="collapsed",
            help="Cada dominio tiene sus propios datos, su propia línea base y su propio estado "
                 "de desarrollo. Lo que cambia entre ellos no es la disciplina — un pronóstico no "
                 "vale nada hasta que le gana a una línea base fijada de antemano — sino qué es "
                 "esa línea base y si ya está construida.",
        )
        module_name, state = DOMAINS[choice]
        st.caption(state)
        st.divider()

    # Imported here, not at module scope: Baloto's page pulls in statsforecast
    # and xgboost, and the cycling page has no use for either.
    importlib.import_module(module_name).render()


main()
