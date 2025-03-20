import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('day_wise.csv')
df['Date'] = pd.to_datetime(df['Date'])

def plot_time_series(start_date, end_date):
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    filtered_df = df.loc[mask]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot the number of new cases over time
    ax1.plot(filtered_df['Date'], filtered_df['New cases'], label='New Cases', color='blue')
    ax1.set_title('New Cases Over Time')
    ax1.set_ylabel('Number of New Cases')
    ax1.legend()

    # Plot the number of deaths over time
    ax2.plot(filtered_df['Date'], filtered_df['New deaths'], label='Deaths', color='red')
    ax2.set_title('Deaths Over Time')
    ax2.set_ylabel('Number of Deaths')
    ax2.legend()

    # Plot the number of recovered people over time
    ax3.plot(filtered_df['Date'], filtered_df['New recovered'], label='Recovered', color='green')
    ax3.set_title('Recovered Over Time')
    ax3.set_ylabel('Number of Recovered')
    ax3.legend()

    ax3.set_xlabel('Date')
    plt.tight_layout()
    plt.show()

start_date = '2020-02-01'
end_date = '2020-04-01'
plot_time_series(start_date, end_date)


# SIR model initial conditions and parameters
S0 = 17000000
I0 = df['Active'].iloc[0]
R0 = df['Recovered'].iloc[0]
D0 = df['Deaths'].iloc[0]
N = S0 + I0 + R0 + D0
alpha = 0.01
beta = 0.3
gamma = 0.1
mu = 0.02

S, I, R, D = [S0], [I0], [R0], [D0]

for t in range(1, len(df)):
    delta_S = alpha * R[-1] - beta * S[-1] * I[-1] / N
    delta_I = beta * S[-1] * I[-1] / N - mu * I[-1] - gamma * I[-1]
    delta_R = gamma * I[-1] - alpha * R[-1]
    delta_D = mu * I[-1]
    
    S.append(S[-1] + delta_S)
    I.append(I[-1] + delta_I)
    R.append(R[-1] + delta_R)
    D.append(D[-1] + delta_D)

# SIR model plot
plt.figure(figsize=(12, 8))
plt.plot(df['Date'], S, label='Susceptible')
plt.plot(df['Date'], I, label='Infected')
plt.plot(df['Date'], R, label='Recovered')
plt.plot(df['Date'], D, label='Deceased')
plt.xlabel('Date')
plt.ylabel('Number of Individuals')
plt.title('SIR Model with Deaths')
plt.legend()
plt.show()
