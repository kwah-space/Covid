import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to simulate the SIR model with deaths
def sir_model(S0, I0, R0, D0, beta, gamma, alpha, mu, days):
    N = S0 + I0 + R0 + D0  # Total population
    S, I, R, D = [S0], [I0], [R0], [D0]
    
    for t in range(1, days):
        dS = alpha * R[-1] - beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - mu * I[-1] - gamma * I[-1]
        dR = gamma * I[-1] - alpha * R[-1]
        dD = mu * I[-1]
        
        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
        D.append(D[-1] + dD)
    
    return S, I, R, D

# Streamlit UI
st.title("SIR Model with Deaths - COVID-19 Simulation")
# tab1, tab2, tab3 = st.tabs("General Information", "SIR Model", "Results")

# Sidebar inputs
st.sidebar.header("Model Parameters")
S0 = st.sidebar.number_input("Initial Susceptible (S0)", 0, 100000000, 17000000, step=1000)
I0 = st.sidebar.number_input("Initial Infected (I0)", 0, 1000000, 1000, step=10)
R0 = st.sidebar.number_input("Initial Recovered (R0)", 0, 1000000, 0, step=10)
D0 = st.sidebar.number_input("Initial Deceased (D0)", 0, 1000000, 0, step=10)

beta = st.sidebar.slider("Beta (Infection Rate)", 0.0, 1.0, 0.3, 0.01)
gamma = st.sidebar.slider("Gamma (Recovery Rate)", 0.0, 1.0, 0.1, 0.01)
alpha = st.sidebar.slider("Alpha (Loss of Immunity Rate)", 0.0, 1.0, 0.05, 0.01)
mu = st.sidebar.slider("Mu (Mortality Rate)", 0.0, 0.1, 0.01, 0.001)

days = st.sidebar.slider("Simulation Days", 1, 365, 180, 1)

# Compute SIR Model
S, I, R, D = sir_model(S0, I0, R0, D0, beta, gamma, alpha, mu, days)

df = pd.DataFrame({"Days": range(days), "Susceptible": S, "Infected": I, "Recovered": R, "Deceased": D})

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Days"], df["Susceptible"], label="Susceptible", color='blue')
ax.plot(df["Days"], df["Infected"], label="Infected", color='red')
ax.plot(df["Days"], df["Recovered"], label="Recovered", color='green')
ax.plot(df["Days"], df["Deceased"], label="Deceased", color='black')
ax.set_xlabel("Days")
ax.set_ylabel("Population")
ax.legend()
st.pyplot(fig)

# Show R0 calculation
R0_value = beta / gamma if gamma > 0 else 0
st.sidebar.markdown(f"**Basic Reproduction Number (R₀):** {R0_value:.2f}")
