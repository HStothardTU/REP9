import streamlit as st
import pandas as pd

st.title('REHIP Model Explorer')

uploaded_file = st.file_uploader('Upload REHIP Model Excel file', type=['xlsx'])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    st.write('Sheets:', xls.sheet_names)
    input_df = pd.read_excel(xls, sheet_name='Input data')
    st.subheader('Input Data (Editable)')
    # Filter main vehicle rows for editing
    vehicle_rows = input_df['Technology'].isin([
        'Bus', 'Car', 'Heavy goods vehicles', 'Light good vehicles', 'Motorcycle'
    ])
    editable_df = input_df[vehicle_rows].copy()
    edited_df = st.data_editor(editable_df, num_rows="dynamic", key="input_editor")
    st.subheader('Calculation Example: Total Distance')
    if 'Total distance [Mmiles]' in edited_df.columns:
        total_distance = edited_df['Total distance [Mmiles]'].sum()
        st.metric('Sum of Total Distance [Mmiles]', f"{total_distance:,.2f}")
    else:
        st.warning('Total distance column not found.')
    st.subheader('Model Calculations (Placeholder)')
    st.write('Model calculations will appear here.')
else:
    st.info('Please upload the REHIP Model Excel file to begin.') 