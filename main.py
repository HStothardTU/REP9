import streamlit as st
import pandas as pd
import numpy as np
import pulp

# Add Net Zero logo to sidebar and header
st.logo(
    "https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png",
    link="https://www.tees.ac.uk/minisites/netzero/index.cfm",
    size="large"
)

# --- Branding/Header Section ---
"""
Branding/Header Section:
Displays the Net Zero logo and project title at the top of the app using HTML for centering and styling.
This reinforces project identity and provides a professional look.
"""
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

years = list(range(2022, 2051))

# --- Editable Tables for All Sub-sectors ---
"""
Editable Tables Section:
Provides editable tables for Road, Rail, and Air sub-sectors using st.data_editor.
Each table allows users to input or adjust miles traveled, fuel shares, costs, and emission factors for each year.
These tables are the core data input for scenario analysis and projections.
"""
def make_default_df(subsector):
    if subsector == 'road':
        return pd.DataFrame({
            "Year": years,
            "Miles Traveled": [1_000_000]*len(years),
            "Fuel Share Petrol": [0.5]*len(years),
            "Fuel Share Diesel": [0.3]*len(years),
            "Fuel Share EV": [0.2]*len(years),
            "Cost per Mile": [0.10]*len(years),
            "CO2 per Mile": [0.2]*len(years),
            "Occupancy": [1.5]*len(years),
        })
    elif subsector == 'rail':
        return pd.DataFrame({
            "Year": years,
            "Miles Traveled": [100_000]*len(years),
            "Fuel Share Diesel": [0.7]*len(years),
            "Fuel Share Electric": [0.3]*len(years),
            "Cost per Mile": [0.08]*len(years),
            "CO2 per Mile": [0.15]*len(years),
            "Occupancy": [50]*len(years),
        })
    elif subsector == 'air':
        return pd.DataFrame({
            "Year": years,
            "Miles Traveled": [50_000]*len(years),
            "Fuel Share Kerosene": [1.0]*len(years),
            "Cost per Mile": [0.5]*len(years),
            "CO2 per Mile": [0.5]*len(years),
            "Occupancy": [120]*len(years),
        })

st.header('Editable Inputs by Sub-sector')
road_df = st.data_editor(make_default_df('road'), num_rows="dynamic", key="road_editor")
rail_df = st.data_editor(make_default_df('rail'), num_rows="dynamic", key="rail_editor")
air_df = st.data_editor(make_default_df('air'), num_rows="dynamic", key="air_editor")

# --- Scenario Save/Load Logic ---
"""
Scenario Save/Load Section:
Allows users to save the current set of input tables as a named scenario, and load any saved scenario for further editing or analysis.
Scenarios are stored in st.session_state['scenarios'] as a dictionary mapping scenario names to their data.
This enables flexible scenario management and comparison.
"""
if 'scenarios' not in st.session_state:
    st.session_state['scenarios'] = {}

scenario_name = st.text_input("Scenario Name", value="Base Scenario")

col1, col2 = st.columns(2)
with col1:
    if st.button("Save Scenario"):
        st.session_state['scenarios'][scenario_name] = {
            "road": road_df.copy(),
            "rail": rail_df.copy(),
            "air": air_df.copy(),
        }
        st.success(f"Scenario '{scenario_name}' saved!")
with col2:
    scenario_options = list(st.session_state['scenarios'].keys())
    selected_scenario = st.selectbox("Load Scenario", options=[""] + scenario_options)
    if selected_scenario and selected_scenario in st.session_state['scenarios']:
        loaded = st.session_state['scenarios'][selected_scenario]
        road_df = loaded["road"]
        rail_df = loaded["rail"]
        air_df = loaded["air"]
        st.info(f"Loaded scenario: {selected_scenario}")

# --- Backcasting Calculation and UI ---
"""
Backcasting Section:
Lets users set a target for total CO₂ emissions in 2050 and calculates the required annual percentage change in emissions to reach that target from the 2022 baseline.
This helps users understand the scale of change needed to meet long-term goals.
"""
st.header("Backcasting: Set End Goals and See Required Changes")
backcast_mode = st.checkbox("Enable Backcasting")
if backcast_mode:
    target_emissions = st.number_input("Target total CO₂ emissions in 2050 (all sub-sectors)", value=0.0)
    # Calculate total emissions for 2022 and 2050
    total_emissions_2022 = (
        road_df.loc[road_df["Year"] == 2022, "Miles Traveled"].values[0] * road_df.loc[road_df["Year"] == 2022, "CO2 per Mile"].values[0] +
        rail_df.loc[rail_df["Year"] == 2022, "Miles Traveled"].values[0] * rail_df.loc[rail_df["Year"] == 2022, "CO2 per Mile"].values[0] +
        air_df.loc[air_df["Year"] == 2022, "Miles Traveled"].values[0] * air_df.loc[air_df["Year"] == 2022, "CO2 per Mile"].values[0]
    )
    n_years = 2050 - 2022
    # Required annual % change in emissions to reach target
    if total_emissions_2022 > 0 and target_emissions >= 0:
        required_pct_change = ( (target_emissions / total_emissions_2022) ** (1/n_years) - 1 ) * 100
        st.write(f"Required annual % change in total emissions: {required_pct_change:.2f}%")
    else:
        st.warning("Check your input values for emissions and target.")

# --- Example Output Table ---
"""
Summary Output Section:
Calculates and displays a table of total emissions by year, summing across all sub-sectors.
This provides a quick overview of the emissions trajectory for the current scenario.
"""
st.header("Summary Table: Total Emissions by Year")
total_emissions = [
    road_df.loc[road_df["Year"] == y, "Miles Traveled"].values[0] * road_df.loc[road_df["Year"] == y, "CO2 per Mile"].values[0] +
    rail_df.loc[rail_df["Year"] == y, "Miles Traveled"].values[0] * rail_df.loc[rail_df["Year"] == y, "CO2 per Mile"].values[0] +
    air_df.loc[air_df["Year"] == y, "Miles Traveled"].values[0] * air_df.loc[air_df["Year"] == y, "CO2 per Mile"].values[0]
    for y in years
]
emissions_df = pd.DataFrame({"Year": years, "Total Emissions": total_emissions})
st.dataframe(emissions_df)

# --- Optimization Logic Section ---
"""
Optimization Logic Section:
Loads the 'Model' sheet from the Excel file, sets up the LP problem using PuLP, defines decision variables, objective, and constraints, and solves the model.
Displays the optimization status and objective value in the Streamlit app.
"""
st.header("Optimization Model (from Excel Sheet)")

uploaded_file = st.file_uploader("Upload REHIP Model Excel file for Optimization", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    model_df = xls.parse("Model", header=None)

    # Create the LP problem
    model = pulp.LpProblem("Minimize_Total_Cost", pulp.LpMinimize)

    # Define decision variables
    rows = range(67, 101)
    cols = range(2, 31)
    variables = {(i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0) for i in rows for j in cols}

    # Objective: Minimize B398 (value shown in sheet)
    objective_value = model_df.iloc[397, 1]
    model += objective_value, "Total_Cost_Objective"

    # Constraint 1: Sum of row 7 values (B7 to AD7) <= B392
    row7_values = model_df.iloc[6, 1:30].fillna(0).values
    constraint_rhs = model_df.iloc[391, 1]
    model += pulp.lpSum(row7_values) <= constraint_rhs, "Row7_Constraint"

    # Block constraints (element-wise ≤)
    row_pairs = [(181, 439), (231, 446), (281, 453), (331, 460), (131, 432)]
    for idx, (lhs_start, rhs_start) in enumerate(row_pairs):
        lhs_block = model_df.iloc[lhs_start:lhs_start+6, 2:31].fillna(0).values
        rhs_block = model_df.iloc[rhs_start:rhs_start+6, 2:31].fillna(0).values
        for i in range(6):
            for j in range(29):
                model += lhs_block[i][j] <= rhs_block[i][j], f"Block_{idx}_r{i}_c{j}"

    # Equality constraints: each row sum = 100 for rows 110–114
    for row in range(109, 114):
        values = model_df.iloc[row, 2:31].fillna(0).values
        model += pulp.lpSum(values) == 100, f"Equality_Row_{row+1}"

    # Solve the model
    model.solve()

    st.write("**Optimization Status:**", pulp.LpStatus[model.status])
    st.write("**Objective value:**", pulp.value(model.objective))
    # Optionally, display decision variables
    # for var in model.variables():
    #     st.write(var.name, "=", var.varValue)
else:
    st.info("Upload the REHIP Model Excel file to run the optimization model.")


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
