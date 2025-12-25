import streamlit as st
# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go


def sirsv_model(t, y, beta, gamma, omega, eta, mu, N=1.0):
    S, I, Rco, Rsolo = y
    dSdt = mu*N + omega*(Rco+Rsolo) - beta*S*I - eta*S - mu*S
    dIdt = beta*S*I - gamma*I - mu*I
    dRcodt = gamma*I - mu*Rco - omega*Rco
    dRsolodt = eta*S - mu*Rsolo - omega*Rsolo
    return [dSdt, dIdt, dRcodt, dRsolodt]


st.set_page_config(
    page_title='SIR + co-offending',
    page_icon='🦠', layout="centered")
st.title('SIR + co-offending')

st.sidebar.subheader('Set-up:')

beta = st.sidebar.slider(
    label='beta', min_value=0.01,
    max_value=1., value=1., step=0.01,
    help='Transmission rate')

gamma = st.sidebar.slider(
    label='gamma', min_value=0.01,
    max_value=1., value=0.1, step=0.01,
    help='Recovery rate')

omega = st.sidebar.slider(
    label='omega', min_value=0.,
    max_value=1., value=0.05, step=0.01,
    help='Rate of skill loss')

eta = st.sidebar.slider(
    label='eta', min_value=0.0,
    max_value=1., value=0.2, step=0.01,
    help='Rate of solo offending')

mu = st.sidebar.slider(
    label='mu', min_value=0.0,
    max_value=1., value=0.04, step=0.01,
    help='Demographic turnover rate')

T = st.sidebar.slider(
    label='T', min_value=50,
    max_value=1000, value=100, step=50,
    help='Number of time steps')

st.sidebar.markdown('---')
st.sidebar.markdown('Initial conditions:')

# Initial conditions
S0 = st.sidebar.slider(
    label='S0', min_value=0.0,
    max_value=1.0, value=0.9, step=0.05,
    help='Initial fraction of naive offenders')

I0 = 1 - S0
st.sidebar.markdown(f'Initial fraction of offenders involved in co-offending: **{I0:.2f}**')



Rc0 = 0.0
Rs0 = 0.0
y0 = [S0, I0, Rc0, Rs0]

# solve ODE
solution = solve_ivp(
    sirsv_model,
    [0, T],
    y0,
    args=(beta, gamma, omega, eta, mu),
    dense_output=False,
    rtol=1e-8,
    atol=1e-10,
)
t = solution.t.round(3)
S, I, Rco, Rsolo = solution.y.round(3)[:]

# Plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(t, S, label='Naive offenders')
# ax.plot(t, I, label='Offenders involved in co-offending')
# ax.plot(t, Rco+Rsolo, color='green', label='Knowledgeable offenders')
# ax.plot(t, Rco, color='green', label='Knowledgeable offenders (co-offending)', lw=1, ls='--')
# ax.plot(t, Rsolo, color='green', label='Knowledgeable offenders (individual)', lw=1, ls='dotted')
# ax.set_ylim(-0.01, 1.01)
# ax.set_xlabel('Time')
# ax.set_ylabel('Proportion')
# ax.set_title("Model dynamics over time", fontweight='bold', loc='left')
# ax.legend()
# st.pyplot(fig)

# plot using plotly

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Naive offenders', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Offenders involved in co-offending', line=dict(color='orange')))
fig2.add_trace(go.Scatter(x=t, y=Rco+Rsolo, mode='lines', name='Knowledgeable offenders', line=dict(color='green')))
fig2.add_trace(go.Scatter(x=t, y=Rco, mode='lines', name='Knowledgeable offenders (co-offending)', line=dict(color='green', dash='dash')))
fig2.add_trace(go.Scatter(x=t, y=Rsolo, mode='lines', name='Knowledgeable offenders (individual)', line=dict(color='green', dash='dot')))
fig2.update_layout(
    title="Model dynamics over time",
    xaxis_title="Time",
    yaxis_title="Proportion",
    yaxis=dict(range=[-0.01, 1.01]),
    legend_title="Compartments"
)
st.plotly_chart(fig2)
