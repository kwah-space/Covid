import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# load data
conn = sqlite3.connect('Data/covid_database.db')
query = """
    SELECT *
    FROM usa_county_wise;
"""

usa_county_df = pd.read_sql(query, conn)
usa_county_df['Date'] = pd.to_datetime(usa_county_df['Date'])
conn.close()

country_daywise_df = pd.read_csv("Data/final_country_daywise.csv")

# Streamlit UI
st.set_page_config(layout="wide")

st.title("🦠 COVID-19 Data Dashboard")
tab1, tab2, tab3 = st.tabs(["🌍 Global Overview", "📊 Country Insights", "🏡 US County Focus"])

# Sidebar for date selection
st.sidebar.header("Customize Your Dashboard")

min_date = country_daywise_df["Date"].min()
max_date = country_daywise_df["Date"].max()

if 'start_date' not in st.session_state:
    st.session_state.start_date = min_date

if 'end_date' not in st.session_state:
    st.session_state.end_date = max_date

start_date = st.sidebar.date_input("Start Date", value=st.session_state.start_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=st.session_state.end_date, min_value=min_date, max_value=max_date)

st.session_state.start_date = start_date
st.session_state.end_date = end_date

# Reset button to clear date range
if st.sidebar.button("Reset Date Range", help="Clear the selected date range by double-clicking"):
    st.session_state.start_date = min_date
    st.session_state.end_date = max_date


sorted_country = country_daywise_df.sort_values(by='Confirmed', ascending=False)['Country.Region'].unique()
selected_country = st.sidebar.selectbox("Select a Country", sorted_country, help="Use for Country Insights")

sorted_counties = usa_county_df.sort_values(by='Confirmed', ascending=False)['Combined_Key'].unique()
selected_uscounty = st.sidebar.selectbox("Select a US County", sorted_counties, help="Use for US County Focus")
    
def get_global_data(df):
    df["Date"] = pd.to_datetime(df["Date"])

    global_data = df.groupby('Date', as_index=False).agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum',
        'Population': 'sum' 
    })

    global_data['New Cases'] = global_data['Confirmed'].diff().fillna(0)
    global_data['New Deaths'] = global_data['Deaths'].diff().fillna(0)
    global_data['New Recovered'] = global_data['Recovered'].diff().fillna(0)

    return global_data

def plot_new_cases_by_country(df, country, start_date=None, end_date=None):
    df = df[df['Country.Region'] == country].copy()

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        subplot_titles=('New Cases', 'New Deaths', 'New Recoveries'),
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['New Cases'], 
                             mode='lines', 
                             name='New Cases'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['New Deaths'], 
                             mode='lines', 
                             name='New Deaths'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['New Recovered'], 
                             mode='lines', 
                             name='New Recoveries'),
                  row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f'New Cases, Deaths, and Recoveries for {country} over Time',
        xaxis_title='Date',
        showlegend=False,
        height=900,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def plot_new_cases_world(df, start_date=None, end_date=None):
    world_data = df.copy()

    world_data['Date'] = pd.to_datetime(world_data['Date'])

    if start_date:
        world_data = world_data[world_data['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        world_data = world_data[world_data['Date'] <= pd.to_datetime(end_date)]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        subplot_titles=('New Cases', 'New Deaths', 'New Recoveries'),
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=world_data['Date'], 
                             y=world_data['New Cases'], 
                             mode='lines', 
                             name='New Cases'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=world_data['Date'], 
                             y=world_data['New Deaths'], 
                             mode='lines', 
                             name='New Deaths'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=world_data['Date'], 
                             y=world_data['New Recovered'], 
                             mode='lines', 
                             name='New Recoveries'),
                  row=3, col=1)

    fig.update_layout(
        title='New Cases, Deaths, and Recoveries Over Time',
        xaxis_title='Date',
        showlegend=False,
        height=800,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

def plot_total_by_country_per_1M(df, country, start_date=None, end_date=None):
    df = country_daywise_df[country_daywise_df['Country.Region'] == country]

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        subplot_titles=('Active Cases per 1M Population', 
                        'Deaths per 1M Population', 
                        'Recovered Cases per 1M Population'),
                vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Active_per_1Mpopulation'], 
                             mode='lines', 
                             name='Active Cases per 1M'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Deaths_per_1Mpopulation'], 
                             mode='lines', 
                             name='Deaths per 1M'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Recovered_per_1Mpopulation'], 
                             mode='lines', 
                             name='Recovered Cases per 1M'),
                  row=3, col=1)

    fig.update_layout(
        title=f'Total Active, Deaths, and Recoveries per 1M Population for {country}',
        xaxis_title='Date',
        showlegend=False,
        height=900,
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Active Cases per 1M", row=1, col=1)
    fig.update_yaxes(title_text="Deaths per 1M", row=2, col=1)
    fig.update_yaxes(title_text="Recovered per 1M", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
def plot_compare_country(df):
    latest_data = df[df["Date"] == df["Date"].max()]

    # Top 5 countries with most cases and deaths
    top_5_cases = latest_data.nlargest(5, "Confirmed")[["Country.Region", "Confirmed"]]
    top_5_deaths = latest_data.nlargest(5, "Deaths")[["Country.Region", "Deaths"]]

    fig_width = 600
    fig_height = 400

    fig_cases = px.bar(top_5_cases, 
                       x="Country.Region", 
                       y="Confirmed", 
                       title="Top 5 Countries with Most Cases", 
                       labels={"Confirmed": "Confirmed"}, 
                       color="Confirmed",
                       color_continuous_scale=px.colors.sequential.Agsunset_r)
    
    fig_cases.update_layout(width=fig_width, height=fig_height)  
    fig_cases.update_layout(coloraxis_showscale=False)  
    fig_cases.update_xaxes(title_text="Country")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_cases, use_container_width=False)  

    # Plot Top 5 Countries with Most Deaths
    fig_deaths = px.bar(top_5_deaths, 
                        x="Country.Region", 
                        y="Deaths", 
                        title="Top 5 Countries with Most Deaths", 
                        labels={"Deaths": "Deaths"}, 
                        color="Deaths",
                       color_continuous_scale=px.colors.sequential.Agsunset_r)
    fig_deaths.update_xaxes(title_text="Country")
    fig_deaths.update_layout(width=fig_width, height=fig_height)
    fig_deaths.update_layout(coloraxis_showscale=False) 
    with col2: 
        st.plotly_chart(fig_deaths, use_container_width=False) 

def plot_map_active(df, continent_select):

    df['ActivePerPopulation'] = df['Active'] / df['Population']

    fig = px.choropleth(df, 
                        locations='Country.Region', 
                        locationmode='country names',
                        color='ActivePerPopulation', 
                        animation_frame='Date',  # Add time-based animation
                        scope=continent_select if continent_select != 'world' else None, 
                        color_continuous_scale=px.colors.sequential.Agsunset_r,
                        labels={'ActivePerPopulation': 'Cases/Population'},
                        title=f'Active COVID-19 Cases per Population in {continent_select} Over Time')

    st.plotly_chart(fig, use_container_width=True)

def plot_death_rates_continent(df):
    continent_df = df.groupby(['Date','Continent'])[['New Deaths','Active']].sum()
    continent_df['Mortality Rate'] = continent_df['New Deaths'] / continent_df['Active']
    continent_death_rate = continent_df.groupby('Continent')['Mortality Rate'].mean()

    continent_death_rate = continent_death_rate.sort_values(ascending=False)

    fig = px.bar(continent_death_rate, 
                x=continent_death_rate.index, 
                y=continent_death_rate.values, 
                labels={'x': 'Continent', 'y': 'Average Mortality Rate'},
                title="Average Mortality Rate per Continent",
                color=continent_death_rate.values,
                color_continuous_scale=px.colors.sequential.Agsunset_r)
    
    fig.update_layout(coloraxis_showscale=False)  
    st.plotly_chart(fig, use_container_width=True)

def calculate_mortality_rate(df):
    df['Mu'] = df['New Deaths'] / df['Active']
    return df

def estimate_parameters(df, gamma):
    N = df['Population'].values[0]  
    alpha = []
    beta = []

    for t in range(1, len(df)):
        S = N - df['Confirmed'].iloc[t] - df['Recovered'].iloc[t] - df['Deaths'].iloc[t]
        I = df['Active'].iloc[t]
        R = df['Recovered'].iloc[t]
        delta_I = df['Active'].iloc[t] - df['Active'].iloc[t-1]
        delta_R = df['Recovered'].iloc[t] - df['Recovered'].iloc[t-1]
        alpha_t = ((gamma * I) - delta_R) / R if R != 0 else np.nan
        beta_t = (N / (S * I)) * (delta_I + (df['Mu'].iloc[t] * I) + (gamma * I))

        alpha.append(alpha_t)
        beta.append(beta_t)

    df['Alpha'] = [np.nan] + alpha 
    df['Beta'] = [np.nan] + beta 

    return df

def compute_R0(df, gamma):
    df['R0'] = df['Beta'] / gamma
    return df

def plot_parameters(df):
    df = calculate_mortality_rate(df)
    gamma = 1/4.5
    df = estimate_parameters(df, gamma)
    df = compute_R0(df, gamma)

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Alpha", "Beta", "Mu", "R0"])

    parameters = ["Alpha", "Beta", "Mu", "R0"]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for param, (row, col) in zip(parameters, positions):
        fig.add_trace(go.Scatter(x=df["Date"], y=df[param], mode="lines", name=param), row=row, col=col)

    fig.update_layout(
        title="Estimated Parameters Over Time",
        template="plotly_white",
        showlegend=False,  
        height=600, width=800
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Parameter Value")

    st.plotly_chart(fig, use_container_width=True)

    alpha = df["Alpha"].mean() if df["Alpha"].mean() < 2  else 1.5

    return alpha, df['Beta'].mean(), df['Mu'].mean()

def sir_model(df, beta, gamma, alpha, mu):
    N = df['Population'].values[0]  
    I0 = df['Active'].values[0]  
    R0 = df['Recovered'].values[0]  
    D0 = df['Deaths'].values[0]  
    S0 = N - I0 - R0 - D0  

    S, I, R, D = [S0], [I0], [R0], [D0]

    for t in range(1, len(df)):
        dS = alpha * R[-1] - beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - mu * I[-1] - gamma * I[-1]
        dR = gamma * I[-1] - alpha * R[-1]
        dD = mu * I[-1]
        
        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
        D.append(D[-1] + dD)
    
    return S, I, R, D

def plot_sir_model(df, S, I, R, D):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Date"], y=S, mode="lines", name="Susceptible"))
    fig.add_trace(go.Scatter(x=df["Date"], y=I, mode="lines", name="Infected"))
    fig.add_trace(go.Scatter(x=df["Date"], y=R, mode="lines", name="Recovered"))
    fig.add_trace(go.Scatter(x=df["Date"], y=D, mode="lines", name="Deceased"))

    fig.update_layout(
        title="SIR Model Over Time",
        xaxis_title="Date",
        yaxis_title="Population",
        template="plotly_white",
        legend_title="Categories"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_new_cases_by_county(county_df, start_date=None, end_date=None):
    df = county_df.copy()
    df['New Deaths'] = df['Deaths'].diff().fillna(0) 
    df['New Cases'] = df['Confirmed'].diff().fillna(0) 

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        subplot_titles=('New Cases', 'New Deaths'),
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['New Cases'], 
                             mode='lines', 
                             name='New Cases'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['New Deaths'], 
                             mode='lines', 
                             name='New Deaths'),
                  row=2, col=1)

    fig.update_layout(
        title=f'New Cases, Deaths over Time',
        xaxis_title='Date',
        showlegend=False,
        height=700,
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def plot_compare_uscounty(df):
    latest_data = df[df["Date"] == df["Date"].max()]

    top_5_cases = latest_data.nlargest(5, "Confirmed")[["Combined_Key", "Confirmed"]]
    top_5_deaths = latest_data.nlargest(5, "Deaths")[["Combined_Key", "Deaths"]]

    fig_width = 600
    fig_height = 400

    fig_cases = px.bar(top_5_cases, 
                       x="Combined_Key", 
                       y="Confirmed", 
                       title="Top 5 US Counties with Most Cases", 
                       labels={"Confirmed": "Confirmed"}, 
                       color="Confirmed",
                       color_continuous_scale=px.colors.sequential.Agsunset_r)
    
    fig_cases.update_layout(width=fig_width, height=fig_height)  
    fig_cases.update_layout(coloraxis_showscale=False)  

    fig_cases.update_xaxes(title_text="County")

    col1, col2 = st.columns(2)
    with col1:
    
        st.plotly_chart(fig_cases, use_container_width=False)  

    fig_deaths = px.bar(top_5_deaths, 
                        x="Combined_Key", 
                        y="Deaths", 
                        title="Top 5 US Counties with Most Deaths", 
                        labels={"Deaths": "Deaths"}, 
                        color="Deaths",
                       color_continuous_scale=px.colors.sequential.Agsunset_r)
    fig_deaths.update_xaxes(title_text="County")
    fig_deaths.update_layout(width=fig_width, height=fig_height)
    fig_deaths.update_layout(coloraxis_showscale=False)  
    with col2:
        st.plotly_chart(fig_deaths, use_container_width=False) 


with tab1:
    st.subheader(f"Overall Statistics")

    df = get_global_data(country_daywise_df)

    total_confirmed = int(df['Confirmed'].iloc[-1])
    total_deaths = int(df["Deaths"].iloc[-1])
    total_recovered = int(df["Recovered"].iloc[-1])
    death_rate = (total_deaths / total_confirmed) * 100
    recovery_rate = (total_recovered / total_confirmed) * 100
    death_rate_str = f"({death_rate:.2f}%)"  
    recovery_rate_str = f"({recovery_rate:.2f}%)"  

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Confirmed Cases", f"{total_confirmed:,}")
    with col2:
        st.metric("Total Deaths (Global Death Rate)", f"{total_deaths:,} {death_rate_str}") 
    with col3:
        st.metric("Total Recovered (Global Recovery Rate)", f"{total_recovered:,} {recovery_rate_str}")

    if 'show_graph' not in st.session_state:
        st.session_state.show_graph = False
    if st.button("Show/Hide Global Trends"):
        st.session_state.show_graph = not st.session_state.show_graph  
    if st.session_state.show_graph:
        plot_new_cases_world(df, start_date, end_date) 
    
    st.subheader(f"Comparison of Different Continents")
    continent_select = st.selectbox('Select a Continent', 
                                    ['africa', 'asia', 'europe', 'north america', 'south america', 'world'])
    
    col1, col2 = st.columns(2)
    with col1:
        plot_map_active(country_daywise_df, continent_select)
    
    with col2:
        plot_death_rates_continent(country_daywise_df)
        
    st.subheader(f"Comparison of Different Countries")
    plot_compare_country(country_daywise_df)


    st.subheader(f"Comparison of Different US Counties")
    plot_compare_uscounty(usa_county_df)


with tab2:
    filtered_df = country_daywise_df[country_daywise_df["Country.Region"] == selected_country]

    st.subheader(f"Statistics for {selected_country}")

    # Calculate metrics
    total_cases = int(filtered_df['Confirmed'].iloc[-1])
    total_deaths = int(filtered_df["Deaths"].iloc[-1])
    total_recovered = int(filtered_df["Recovered"].iloc[-1])
    active_cases = int(filtered_df["Active"].iloc[-1])
    population = int(filtered_df["Population"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confirmed Cases", f"{total_cases:,}")
    with col2:
        st.metric("Active Cases", f"{active_cases:,}")
    with col3:
        st.metric("Total Deaths", f"{total_deaths:,}")
    with col4:
        st.metric("Total Recovered", f"{total_recovered:,}")


    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Population", f"{population:,}")
    with col6:
        st.metric("Active Cases per 1M Population", f"{total_cases / population * 1e6:.0f}")
    with col7:
        st.metric("Deaths per 1M Population", f"{total_deaths / population * 1e6:.0f}")
    with col8:
        st.metric("Recovered per 1M Population", f"{total_recovered / population * 1e6:.0f}")

    col1, col2 = st.columns(2)
    with col1:
        plot_new_cases_by_country(country_daywise_df, selected_country, start_date, end_date)
    with col2:
        plot_total_by_country_per_1M(country_daywise_df, selected_country, start_date, end_date)

    st.subheader(f"SIR Model Summaries for {selected_country}")
    parameters_mean = plot_parameters(filtered_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gamma = st.number_input("Gamma", value=1/14, step=0.005, help="Recovery rate of the infected population", format="%0.3f")
    with col2:
        beta = st.number_input("Beta", value=parameters_mean[1], step=0.005, help="Infection rate per person per day", format="%0.3f")
    with col3:
        alpha = st.number_input("Alpha", value=parameters_mean[0], step=0.01, help="Loss of immunity rate per person per day", format="%0.3f")
    with col4:
        mu = st.number_input("Mu", value=parameters_mean[2], step=0.005, help="Mortality rate per person per day", format="%0.3f")

    plot_sir_model(filtered_df, *sir_model(filtered_df, beta, gamma, alpha, mu))


with tab3:
    st.subheader(f"Statistics for {selected_uscounty}")

    county_df = usa_county_df[usa_county_df["Combined_Key"] == selected_uscounty]

    total_cases = int(county_df['Confirmed'].max())
    total_deaths = int(county_df["Deaths"].max())
    death_rate = (total_deaths / total_cases) * 100
    death_rate_str = f"({death_rate:.2f}%)"  

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confirmed Cases", f"{total_cases:,}")
    with col2:
        st.metric("Total Deaths (Global Death Rate)", f"{total_deaths:,}{death_rate_str}") 

    plot_new_cases_by_county(county_df, start_date, end_date)


# streamlit run streamlit_app.py
