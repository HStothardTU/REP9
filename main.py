import streamlit as st
import pandas as pd

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
