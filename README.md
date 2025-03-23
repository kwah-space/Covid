# COVID-19 Data Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dashboard-kit.streamlit.app/)


## Overview

This repository contains a COVID-19 data dashboard built using Streamlit. The dashboard visualizes the spread and impact of COVID-19 from January 22nd to July 27th, 2020. It provides interactive graphs and insights into confirmed cases, deaths, and recoveries across different countries and regions.


## Features

- Interactive date range selection
- Comparative analysis across different continents, countries, and US counties
- General information of each country and US county
- SIR Model summaries of each country
- Key metrics:
  - Total Confirmed Cases
  - Total Deaths
  - Total Recoveries 
  - Population
- Option to input values for SIR model parameters


## Installation
To run the dashboard locally, follow these steps:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## Requirements

- Python 3.7+
- Streamlit
- Plotly
- Pandas
- NumPy
