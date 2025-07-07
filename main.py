import streamlit as st
import pandas as pd
import numpy as np

# Add Net Zero logo to sidebar and header
st.logo(
    "https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png",
    link="https://www.tees.ac.uk/minisites/netzero/index.cfm",
    size="large"
)

primaryColor = "#6eb52f"         # Net Zero green
backgroundColor = "#ffffff"      # White background
secondaryBackgroundColor = "#f0f0f5"  # Light gray for sidebar
textColor = "#262730"            # Dark gray text
font = "sans serif" 


st.title('REHIP Model Explorer')

uploaded_file = st.file_uploader('Upload REHIP Model Excel file', type=['xlsx'])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    st.write('Sheets:', xls.sheet_names)
    input_df = pd.read_excel(xls, sheet_name='Input data')
    st.subheader('Input Data (Editable)')
    # Select main vehicle rows and key columns
    vehicle_types = [
        'Bus', 'Car', 'Heavy goods vehicles', 'Light good vehicles', 'Motorcycle'
    ]
    key_cols = [
        'Technology', 'Total distance [Mmiles]', 'Diesel', 'Petrol', 'HEV', 'Other fuel', 'EV', 'H2'
    ]
    editable_df = input_df[input_df['Technology'].isin(vehicle_types)][key_cols].copy()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", key="input_editor")
    st.subheader('Core Calculations: Total Fuel Demand by Year (Simple MVP)')
    # Calculate total demand for each fuel type (sum across all vehicles)
    total_petrol = edited_df['Petrol'].sum() if 'Petrol' in edited_df.columns else 0
    total_diesel = edited_df['Diesel'].sum() if 'Diesel' in edited_df.columns else 0
    total_h2 = edited_df['H2'].sum() if 'H2' in edited_df.columns else 0
    total_ev = edited_df['EV'].sum() if 'EV' in edited_df.columns else 0
    st.metric('Total Petrol Demand', f"{total_petrol:,.2f}")
    st.metric('Total Diesel Demand', f"{total_diesel:,.2f}")
    st.metric('Total H2 Demand', f"{total_h2:,.2f}")
    st.metric('Total Electricity Demand (EV)', f"{total_ev:,.2f}")
    st.subheader('Model Calculations (Placeholder)')
    st.write('More detailed model calculations and year-by-year projections will appear here.')
else:
    st.info('Please upload the REHIP Model Excel file to begin.')

years = list(range(2022, 2051))

st.sidebar.header('Scenario Controls: Initial Value and Annual % Change')
# User sets initial value and annual % change for each fuel type
fuel_types = ['Petrol', 'Diesel', 'H2', 'EV']
initials = {}
changes = {}
for fuel in fuel_types:
    initials[fuel] = st.sidebar.number_input(f'Initial {fuel} Demand (2022)', min_value=0.0, value=100.0, step=10.0)
    changes[fuel] = st.sidebar.slider(f'Annual % Change for {fuel}', min_value=-20.0, max_value=20.0, value=0.0, step=0.5)

# Calculate year-by-year demand
def project_demand(initial, pct_change, years):
    vals = [initial]
    for _ in range(1, len(years)):
        vals.append(vals[-1] * (1 + pct_change / 100))
    return vals

results = {}
for fuel in fuel_types:
    results[fuel] = project_demand(initials[fuel], changes[fuel], years)

# Create DataFrame for display
df_proj = pd.DataFrame(results, index=years)

st.subheader('Year-by-Year Fuel Demand Projections (2022–2050)')
st.line_chart(df_proj)
st.dataframe(df_proj.style.format('{:,.2f}'))

st.info('Adjust the initial values and annual % changes in the sidebar to see the effect on year-by-year fuel demand projections.') 

st.sidebar.image("https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png", use_column_width=True)
st.markdown("<h1 style='text-align: center; color: #6eb52f;'>Net Zero Transport Model</h1>", unsafe_allow_html=True)

st.markdown(
    '''
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f0f0f5;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #e0e0ef;
    }
    </style>
    ''',
    unsafe_allow_html=True
) 
