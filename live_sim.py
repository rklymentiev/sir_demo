"""Animated agent-level companion to the compartmental model.

The whole simulation lives inside a single self-contained HTML component.
Its controls are rendered *in the iframe* rather than as Streamlit widgets,
because Streamlit reruns the script on every widget interaction and
``components.html`` remounts the iframe whenever the HTML string changes --
which would restart the animation on every slider nudge.

The component is therefore independent of the sidebar. Use
``sync_from_sidebar`` to push the current sidebar parameters into it on
demand; that intentionally does restart the animation.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_HTML = Path(__file__).with_name("sim_component.html")


def render_live_simulation(params: dict, dark: bool, height: int = 620) -> None:
    """Draw the animated compartment model.

    ``params`` needs beta, gamma, omega, eta, mu, N and S0. Changing any of
    them changes the HTML string, so the component remounts and the
    simulation restarts. Call this with values that only change when the
    user explicitly asks for it.
    """
    html = _HTML.read_text(encoding="utf-8")
    html = html.replace("__THEME__", "dark" if dark else "light")
    html = html.replace("__PARAMS__", json.dumps(params))
    components.html(html, height=height, scrolling=False)


def simulation_tab(sidebar_params: dict, dark: bool) -> None:
    """The full tab: sync control, component, and a short explanation."""
    st.session_state.setdefault(
        "sim_params",
        dict(beta=1.0, gamma=0.1, omega=0.1, eta=0.2, mu=0.05, N=240, S0=0.9),
    )

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            "Each dot is one active offender moving between the three states. "
            "The controls below belong to the animation and are independent of "
            "the sidebar, so adjusting them does not interrupt the run."
        )
    with right:
        if st.button("Load sidebar values", use_container_width=True,
                     help="Copies the sidebar parameters into the animation. "
                          "This restarts the simulation."):
            st.session_state.sim_params = {
                **st.session_state.sim_params,
                **{k: sidebar_params[k]
                   for k in ("beta", "gamma", "omega", "eta", "mu")},
            }

    render_live_simulation(st.session_state.sim_params, dark)
