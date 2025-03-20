import sqlite3

conn = sqlite3.connect('/data/covid_database.db')
cursor = conn.cursor()

def get_column_names(table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return [column[1] for column in columns]

# Get column names for each table
country_wise_columns = get_column_names('country_wise')
day_wise_columns = get_column_names('day_wise')
usa_county_wise_columns = get_column_names('usa_county_wise')
worldometer_data_columns = get_column_names('worldometer_data')

print("Country Wise Columns:", country_wise_columns)
print("Day Wise Columns:", day_wise_columns)
print("USA County Wise Columns:", usa_county_wise_columns)
print("Worldometer Data Columns:", worldometer_data_columns)
