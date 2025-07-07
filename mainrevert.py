import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Add Net Zero logo to sidebar and header
st.logo(
    "https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png",
    link="https://www.tees.ac.uk/minisites/netzero/index.cfm",
    size="large"
)

# Centered logo and subtitle at the top of the main page
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 0.5em;'>
        <img src='assets/netzero-logo.svg' width='220' style='margin-bottom: 0.2em;'>
        <h2 style='color: #6eb52f; margin-bottom: 0.2em;'>Net Zero Transport Model</h2>
        <p style='color: #262730; font-size: 1.1em;'>Scenario Analysis & Fuel Demand Projections</p>
    </div>
    """,
    unsafe_allow_html=True
)

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
fuel_types = ['Petrol', 'Diesel', 'H2', 'EV']

# --- Scenario Management ---
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = {}
if 'scenario_name' not in st.session_state:
    st.session_state['scenario_name'] = 'Base Scenario'

st.sidebar.header('Scenario Controls')
scenario_name = st.sidebar.text_input('Scenario Name', value=st.session_state['scenario_name'])

# User sets initial value, annual % change, and target for each fuel type
initials = {}
changes = {}
targets = {}
for fuel in fuel_types:
    initials[fuel] = st.sidebar.number_input(f'Initial {fuel} Demand (2022)', min_value=0.0, value=100.0, step=10.0, key=f'init_{fuel}')
    changes[fuel] = st.sidebar.slider(f'Annual % Change for {fuel}', min_value=-20.0, max_value=20.0, value=0.0, step=0.5, key=f'chg_{fuel}')
    targets[fuel] = st.sidebar.number_input(f'Target {fuel} Demand (2050)', min_value=0.0, value=100.0, step=10.0, key=f'tgt_{fuel}')

# Save scenario button
if st.sidebar.button('Save Scenario'):
    st.session_state['scenarios'][scenario_name] = {
        'initials': initials.copy(),
        'changes': changes.copy(),
        'targets': targets.copy()
    }
    st.session_state['scenario_name'] = scenario_name
    st.success(f"Scenario '{scenario_name}' saved!")

# Scenario selector
if st.session_state['scenarios']:
    selected_scenarios = st.sidebar.multiselect(
        'Compare Scenarios',
        options=list(st.session_state['scenarios'].keys()),
        default=[scenario_name]
    )
else:
    selected_scenarios = []

# Always include the current (unsaved) scenario as 'Current'
selected_scenarios = ['Current'] + selected_scenarios
scenarios_to_plot = {'Current': {'initials': initials, 'changes': changes, 'targets': targets}}
for name in selected_scenarios:
    if name != 'Current' and name in st.session_state['scenarios']:
        scenarios_to_plot[name] = st.session_state['scenarios'][name]

# --- Projection and Visualization ---
def project_demand(initial, pct_change, years):
    vals = [initial]
    for _ in range(1, len(years)):
        vals.append(vals[-1] * (1 + pct_change / 100))
    return vals

# Build a long DataFrame for Altair
all_proj = []
all_targets = []
all_achieved = []
for scen_name, params in scenarios_to_plot.items():
    initials = params['initials']
    changes = params['changes']
    targets = params['targets']
    for fuel in fuel_types:
        vals = project_demand(initials[fuel], changes[fuel], years)
        arr = np.array(vals)
        idx = np.where(arr >= targets[fuel])[0]
        achieved = years[idx[0]] if len(idx) > 0 else None
        all_proj.append(pd.DataFrame({
            'Year': years,
            'Demand': vals,
            'Fuel': fuel,
            'Scenario': scen_name
        }))
        all_targets.append(pd.DataFrame({
            'Year': years,
            'Target': [targets[fuel]]*len(years),
            'Fuel': fuel,
            'Scenario': scen_name
        }))
        all_achieved.append({'Scenario': scen_name, 'Fuel': fuel, 'Achieved': achieved})
proj_df = pd.concat(all_proj, ignore_index=True)
target_df = pd.concat(all_targets, ignore_index=True)
achieved_df = pd.DataFrame(all_achieved)

st.subheader('Year-by-Year Fuel Demand Projections (2022–2050)')
# Altair chart with scenario comparison
chart = alt.Chart(proj_df).mark_line().encode(
    x='Year:O',
    y='Demand:Q',
    color='Scenario:N',
    strokeDash='Fuel:N',
    tooltip=['Year', 'Fuel', 'Scenario', 'Demand']
)
# Add target lines
target_lines = alt.Chart(target_df).mark_line(strokeDash=[5,5], color='gray').encode(
    x='Year:O',
    y='Target:Q',
    detail='Fuel:N',
    color='Scenario:N'
)
final_chart = chart + target_lines
st.altair_chart(final_chart, use_container_width=True)

# Show projections in a table
st.dataframe(proj_df.pivot_table(index=['Year'], columns=['Scenario', 'Fuel'], values='Demand').round(2))

# Show target achievement info
for scen_name in scenarios_to_plot:
    st.markdown(f"**Scenario: {scen_name}**")
    for fuel in fuel_types:
        achieved = achieved_df[(achieved_df['Scenario'] == scen_name) & (achieved_df['Fuel'] == fuel)]['Achieved'].values[0]
        if achieved is not None:
            st.success(f"{fuel}: Target met in {achieved}")
        else:
            st.warning(f"{fuel}: Target not met by 2050")

st.info('You can define and save multiple scenarios, then compare their projections and target achievement in the chart and table.')


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
