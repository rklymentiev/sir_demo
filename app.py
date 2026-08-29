import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Co-offending dynamics",
    page_icon="👥",
    layout="wide",
)

# palette matched to the manuscript figures
C_S, C_C, C_K = "#1f77b4", "#ff7f0e", "#2ca02c"


# detect the active theme so the plots adapt (st.context.theme needs a
# recent Streamlit; fall back to light styling if unavailable)
try:
    _dark = st.context.theme.type == "dark"
except Exception:
    _dark = False

PLOT_TEMPLATE = "plotly_dark" if _dark else "simple_white"
FG = "#e8e8ea" if _dark else "rgba(120,120,130,0.9)"          # trajectory / foreground line
ARROW = "rgba(190,190,200,0.45)" if _dark else "rgba(120,120,130,0.45)"
PLOT_BG = "rgba(0,0,0,0)"                        # let the page show through

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      div[data-testid="stMetric"] {
          background: rgba(128, 128, 128, 0.10);
          border: 1px solid rgba(128, 128, 128, 0.25);
          border-radius: 8px; padding: 12px 16px;
      }
      section[data-testid="stSidebar"] {width: 340px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Why does co-offending persist?")
st.caption(
    "An interactive companion to the compartmental model of co-offending and "
    "criminal skill exchange."
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def model(t, y, beta, gamma, omega, eta, mu, N=1.0):
    """
    S  naive solo offenders
    Kc knowledgeable via co-offending, Ka knowledgeable via asocial learning
    C  offenders currently co-offending      (K = Kc + Ka)
    """
    S, C, Kc, Ka = y
    dS = mu * N + omega * (Kc + Ka) - beta * S * C - eta * S - mu * S
    dC = beta * S * C - gamma * C - mu * C
    dKc = gamma * C - mu * Kc - omega * Kc
    dKa = eta * S - mu * Ka - omega * Ka
    return [dS, dC, dKc, dKa]


def reproduction_number(beta, gamma, omega, eta, mu):
    denom = (mu + omega + eta) * (gamma + mu)
    if denom == 0:
        return np.nan
    return beta * (mu + omega) / denom


def equilibria(beta, gamma, omega, eta, mu):
    """Returns (COFE_S, EE_S, EE_C). EE values are None when no EE exists."""
    cofe_S = (mu + omega) / (mu + omega + eta) if (mu + omega + eta) > 0 else 1.0
    num = beta * (mu + omega) - (mu + omega + eta) * (gamma + mu)
    den = beta * (gamma + mu + omega)
    if num > 0 and den > 0:
        return cofe_S, (gamma + mu) / beta, num / den
    return cofe_S, None, None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("Parameters")

preset = st.sidebar.selectbox(
    "Preset",
    ["Custom",
     "Endemic co-offending",
     "Co-offending dies out",
     "Baseline model (no decay, turnover or asocial learning)",
     "Skill decay only (μ = η = 0)",
     "Turnover only (ω = η = 0)"],
)

P = dict(beta=1.0, gamma=0.1, omega=0.1, eta=0.2, mu=0.05)
if preset == "Co-offending dies out":
    P = dict(beta=0.1, gamma=1.0, omega=0.05, eta=0.2, mu=0.04)
elif preset.startswith("Baseline"):
    P = dict(beta=1.0, gamma=0.1, omega=0.0, eta=0.0, mu=0.0)
elif preset.startswith("Skill decay only"):
    P = dict(beta=1.0, gamma=0.1, omega=0.1, eta=0.0, mu=0.0)
elif preset.startswith("Turnover only"):
    P = dict(beta=1.0, gamma=0.1, omega=0.0, eta=0.0, mu=0.05)

beta = st.sidebar.slider(
    r"Collaboration-initiation rate ($\beta$)", 0.0, 1.0, P["beta"], 0.01,
    help="Rate at which naive offenders begin co-offending: recruitment, "
         "situational demand for partners, and convergence in space and time.")

gamma = st.sidebar.slider(
    r"Collaborative-learning rate ($\gamma$)", 0.01, 1.0, P["gamma"], 0.01,
    help="Rate at which co-offenders acquire enough skill to offend alone. "
         "Higher values mean shorter reliance on partners.")

omega = st.sidebar.slider(
    r"Skill-decay rate ($\omega$)", 0.0, 1.0, P["omega"], 0.01,
    help="Rate at which knowledgeable offenders lose the skill and return to "
         "the naive state, through forgetting or obsolescence.")

eta = st.sidebar.slider(
    r"Asocial-learning rate ($\eta$)", 0.0, 1.0, P["eta"], 0.01,
    help="Rate at which naive offenders acquire the skill on their own, "
         "bypassing co-offending entirely.")

mu = st.sidebar.slider(
    r"Demographic turnover rate ($\mu$)", 0.0, 0.5, P["mu"], 0.01,
    help="Onset of new naive offenders, and exit through desistance, "
         "incapacitation or death.")

st.sidebar.markdown("---")
T = st.sidebar.slider("Time horizon", 50, 500, 250, 50)
S0 = st.sidebar.slider(
    r"Initial naive share ($S_0$)", 0.0, 1.0, 0.9, 0.05,
    help="The remainder starts in the co-offending state.")
st.sidebar.caption(f"$C_0$ = {1 - S0:.2f},  $K_0$ = 0.00")


# ----------------------------------------------------------------------------
# Solve
# ----------------------------------------------------------------------------
sol = solve_ivp(model, [0, T], [S0, 1 - S0, 0.0, 0.0],
                args=(beta, gamma, omega, eta, mu),
                dense_output=True, rtol=1e-8, atol=1e-10,
                t_eval=np.linspace(0, T, 1200))
t = sol.t
S, C, Kc, Ka = sol.y
K = Kc + Ka

R0 = reproduction_number(beta, gamma, omega, eta, mu)
cofe_S, ee_S, ee_C = equilibria(beta, gamma, omega, eta, mu)

# ----------------------------------------------------------------------------
# Headline readout
# ----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Basic reproduction number $\\mathcal{R}_0$",
          "—" if np.isnan(R0) else f"{R0:.2f}")
m2.metric("Long-run outcome",
          "Endemic (EE)" if (ee_C is not None) else "Dies out (COFE)")
m3.metric("Equilibrium co-offending $C^*$",
          f"{ee_C:.3f}" if ee_C is not None else "0.000")
m4.metric("Co-offending at end of run", f"{C[-1]:.3f}")

if np.isnan(R0):
    st.warning(
        "With $\\mu = \\omega = \\eta = 0$ the model reduces to the baseline "
        "specification and $\\mathcal{R}_0$ is undefined ($0/0$). Skill is "
        "permanent and the naive pool is never replenished, so co-offending "
        "cannot persist however large $\\beta$ is."
    )
elif ee_C is not None:
    st.success(
        f"$\\mathcal{{R}}_0 = {R0:.2f} > 1$. Co-offending is self-sustaining: "
        f"the system settles at an endemic equilibrium with "
        f"{ee_C:.1%} of offenders collaborating."
    )
else:
    st.info(
        f"$\\mathcal{{R}}_0 = {R0:.2f} < 1$. Co-offending cannot sustain "
        "itself and the population converges to the co-offending-free "
        "equilibrium."
    )

tab1, tab2, tab3 = st.tabs(["Dynamics over time", "Phase portrait", "About the model"])

# ----------------------------------------------------------------------------
# Tab 1: time series
# ----------------------------------------------------------------------------
with tab1:
    show_split = st.checkbox(
        "Split knowledgeable offenders by how they learned", value=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, name="Naive solo (S)",
                             line=dict(color=C_S, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=C, name="Co-offending (C)",
                             line=dict(color=C_C, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=K, name="Knowledgeable solo (K)",
                             line=dict(color=C_K, width=2.5)))
    if show_split:
        fig.add_trace(go.Scatter(x=t, y=Kc, name="K via co-offending",
                                 line=dict(color=C_K, width=1.3, dash="dash")))
        fig.add_trace(go.Scatter(x=t, y=Ka, name="K via asocial learning",
                                 line=dict(color=C_K, width=1.3, dash="dot")))
    if ee_C is not None:
        fig.add_hline(y=ee_C, line=dict(color=C_C, width=1, dash="dot"),
                      annotation_text="C*", annotation_position="right")
    fig.update_layout(
        template=PLOT_TEMPLATE, height=460,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis_title="Time", yaxis_title="Proportion of offenders",
        xaxis=dict(hoverformat=".2f"),
        yaxis=dict(range=[-0.02, 1.02], hoverformat=".2f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 2: phase portrait
# ----------------------------------------------------------------------------
with tab2:
    g = np.linspace(0.001, 1, 20)
    SS, CC = np.meshgrid(g, g)
    mask = SS + CC <= 1
    dS = mu + omega * (1 - SS - CC) - beta * SS * CC - eta * SS - mu * SS
    dC = beta * SS * CC - (gamma + mu) * CC
    norm = np.hypot(dS, dC)
    norm[norm == 0] = 1
    sc = 0.045

    fig2 = go.Figure()
    for i in range(SS.shape[0]):
        for j in range(SS.shape[1]):
            if not mask[i, j]:
                continue
            x0, y0 = SS[i, j], CC[i, j]
            fig2.add_annotation(
                x=x0 + sc * dS[i, j] / norm[i, j],
                y=y0 + sc * dC[i, j] / norm[i, j],
                ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1,
                arrowwidth=0.8, arrowcolor=ARROW)

    fig2.add_trace(go.Scatter(x=S, y=C, mode="lines", name="Trajectory",
                              line=dict(color=FG, width=2.5)))
    fig2.add_trace(go.Scatter(x=[S0], y=[1 - S0], mode="markers", name="Start",
                              marker=dict(color=C_K, size=11)))
    # COFE is a continuum of equilibria whenever nothing moves the system
    # along C = 0, i.e. when mu = omega = eta = 0. Any one of them being
    # positive collapses it to a single point.
    cofe_is_line = (mu == 0 and omega == 0 and eta == 0)
    if cofe_is_line:
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 0], mode="lines", name="COFE (continuum)",
            line=dict(color="blue", width=4)))
    else:
        fig2.add_trace(go.Scatter(
            x=[cofe_S], y=[0], mode="markers", name="COFE",
            marker=dict(color="blue", size=13, symbol="diamond")))
    if ee_C is not None:
        fig2.add_trace(go.Scatter(x=[ee_S], y=[ee_C], mode="markers", name="EE",
                                  marker=dict(color="magenta", size=13,
                                              symbol="diamond")))
        fig2.add_vline(x=ee_S, line=dict(color="orange", width=1.5),
                       annotation_text="S = (γ+μ)/β")
    fig2.update_layout(
        template=PLOT_TEMPLATE, height=560,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis_title="Naive solo offenders (S)",
        yaxis_title="Offenders co-offending (C)",
        xaxis=dict(range=[0, 1], hoverformat=".2f"),
        yaxis=dict(range=[0, 1], hoverformat=".2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
    if cofe_is_line:
        st.caption(
            "Arrows show the direction of flow. With $\\mu = \\omega = \\eta = 0$ "
            "nothing moves the system along $C = 0$, so **every** point on that "
            "line is an equilibrium: the COFE is a continuum, and where the "
            "population ends up depends on its starting point. Co-offending "
            "always dies out, but the surviving naive share varies."
        )
    else:
        st.caption(
            "Arrows show the direction of flow. The trajectory converges to the "
            "endemic equilibrium (EE) when $\\mathcal{R}_0 > 1$, and to the "
            "co-offending-free equilibrium (COFE) otherwise. Note that adding "
            "any replenishment or bypass mechanism ($\\mu$, $\\omega$ or "
            "$\\eta$) collapses the baseline model's *line* of co-offending-free "
            "equilibria to this single point."
        )

# ----------------------------------------------------------------------------
# Tab 3: explanation
# ----------------------------------------------------------------------------
with tab3:
    st.markdown(
        r"""
### The model

Active offenders occupy one of three states relative to a criminal skill:

| State | Meaning |
|---|---|
| $S$ | naive solo offenders, who lack the skill |
| $C$ | offenders currently co-offending |
| $K$ | knowledgeable solo offenders, who have the skill |

$$\frac{dS}{dt} = \mu N - \beta S C + \omega K - \eta S - \mu S$$

$$\frac{dC}{dt} = \beta S C - \gamma C - \mu C$$

$$\frac{dK}{dt} = \gamma C - \omega K - \mu K + \eta S$$

### Thresholds

$$\mathcal{R}_0 = \frac{\beta(\mu+\omega)}{(\mu+\omega+\eta)(\gamma+\mu)}$$

An endemic equilibrium exists and is stable when $\mathcal{R}_0 > 1$, with

$$C^* = \frac{\beta(\mu+\omega) - (\mu+\omega+\eta)(\gamma+\mu)}{\beta(\gamma+\mu+\omega)}$$

### Things worth trying

- Set $\mu = \eta = 0$ and raise $\omega$. Co-offending persists on **skill
  decay alone**.
- Set $\omega = \eta = 0$ and raise $\mu$. It persists on **turnover alone**.
  Notice that $C^*$ peaks at intermediate $\mu$ and then falls.
- Set $\mu = \omega = 0$. The model collapses to the baseline case and
  co-offending always dies out, however large $\beta$ is.
- Raise $\eta$. Asocial learning competes with collaboration and pushes
  $\mathcal{R}_0$ down.

Because $\mu$ and $\omega$ enter the numerator only as the sum $(\mu+\omega)$,
either mechanism **on its own** is enough to sustain co-offending. Only when
both are zero does it become impossible.
"""
    )