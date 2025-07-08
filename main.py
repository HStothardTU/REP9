import streamlit as st
import pandas as pd
import numpy as np
import pulp

# --- Calculation Functions (Refactored Backend Logic) ---
"""
Calculation Functions Section:
Provides robust, reusable functions for calculating total CO2 emissions and costs for each transport mode and for totals.

Limitations:
- Teesside data estimation: Data for Teesside is estimated where direct data is unavailable.
- Goods train exclusion: Goods (freight) trains are not included in the calculations.
- Commercial flights only: Only commercial flights are considered for the air sector.
These limitations are reflected in the data processing and should be clearly communicated to users.
"""
def calculate_total_co2_emissions(df, per_passenger_mile=False):
    """
    Calculate total CO2 emissions for a given transport mode DataFrame.
    If per_passenger_mile is True, uses 'CO2 per pg.mile' and 'pg.miles_travelled'.
    Otherwise, uses 'CO2 per Mile' and 'Miles Traveled'.
    """
    if per_passenger_mile:
        if 'CO2 per pg.mile' in df.columns and 'pg.miles_travelled' in df.columns:
            return (df['CO2 per pg.mile'] * df['pg.miles_travelled']).sum()
        else:
            raise ValueError("DataFrame must contain 'CO2 per pg.mile' and 'pg.miles_travelled' columns for per_passenger_mile calculation.")
    else:
        return (df['CO2 per Mile'] * df['Miles Traveled']).sum()

def calculate_total_cost(df, per_passenger_mile=False):
    """
    Calculate total cost for a given transport mode DataFrame.
    If per_passenger_mile is True, uses 'Cost per pg.mile' and 'pg.miles_travelled'.
    Otherwise, uses 'Cost per Mile' and 'Miles Traveled'.
    """
    if per_passenger_mile:
        if 'Cost per pg.mile' in df.columns and 'pg.miles_travelled' in df.columns:
            return (df['Cost per pg.mile'] * df['pg.miles_travelled']).sum()
        else:
            raise ValueError("DataFrame must contain 'Cost per pg.mile' and 'pg.miles_travelled' columns for per_passenger_mile calculation.")
    else:
        return (df['Cost per Mile'] * df['Miles Traveled']).sum()

def calculate_annual_emissions(df, per_passenger_mile=False):
    """
    Returns a Series of total CO2 emissions per year for the given DataFrame.
    """
    if per_passenger_mile:
        return df['CO2 per pg.mile'] * df['pg.miles_travelled']
    else:
        return df['CO2 per Mile'] * df['Miles Traveled']

def calculate_annual_cost(df, per_passenger_mile=False):
    """
    Returns a Series of total cost per year for the given DataFrame.
    """
    if per_passenger_mile:
        return df['Cost per pg.mile'] * df['pg.miles_travelled']
    else:
        return df['Cost per Mile'] * df['Miles Traveled']


# Add Net Zero logo to sidebar and header
st.logo(
    "https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png",
    link="https://www.tees.ac.uk/minisites/netzero/index.cfm",
    size="large"
)

# --- Branding/Header Section ---

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

# --- Scenario Controls: Current Year and 2025 Target Demand ---
"""
Scenario Controls Section:
Allows users to set the current year's demand (from the editable table) and a target demand for 2025 for each sub-sector. When a scenario is created or loaded, the app interpolates demand from the current year to the 2025 target, and uses this as the baseline for projections.
"""
st.header('Scenario Controls: Set 2025 Target Demand')

# Get current year (first year in years list)
current_year = years[0]

# UI for 2025 target demand for each sub-sector
col1, col2, col3 = st.columns(3)
with col1:
    road_2025_target = st.number_input(
        'Road: Target Miles Traveled in 2025',
        value=float(road_df.loc[road_df['Year'] == 2025, 'Miles Traveled'].values[0])
    )
with col2:
    rail_2025_target = st.number_input(
        'Rail: Target Miles Traveled in 2025',
        value=float(rail_df.loc[rail_df['Year'] == 2025, 'Miles Traveled'].values[0])
    )
with col3:
    air_2025_target = st.number_input(
        'Air: Target Miles Traveled in 2025',
        value=float(air_df.loc[air_df['Year'] == 2025, 'Miles Traveled'].values[0])
    )

def interpolate_demand(df, target_2025):
    """Interpolate Miles Traveled from current year to 2025 target, then keep rest as is."""
    df = df.copy()
    y0 = df.loc[df['Year'] == current_year, 'Miles Traveled'].values[0]
    y5 = target_2025
    for y in range(current_year, 2026):
        frac = (y - current_year) / (2025 - current_year) if 2025 != current_year else 0
        df.loc[df['Year'] == y, 'Miles Traveled'] = y0 + frac * (y5 - y0)
    return df

# Apply interpolation to scenario tables before saving/loading
if st.button('Apply 2025 Targets to Scenario'):
    road_df = interpolate_demand(road_df, road_2025_target)
    rail_df = interpolate_demand(rail_df, rail_2025_target)
    air_df = interpolate_demand(air_df, air_2025_target)
    st.success('2025 targets applied to scenario tables!')

# --- Enhanced Scenario Modeling Section ---
st.header("Scenario Modeling: Adjust Parameters and Run What-Ifs")

with st.expander("Advanced Scenario Parameters", expanded=False):
    st.markdown("Adjust fuel shares and adoption rates for each sector.")
    # Example sliders for fuel shares (can be expanded for more detail)
    road_petrol_share = st.slider("Road: Petrol Share (2025)", 0.0, 1.0, float(road_df.loc[road_df['Year'] == 2025, 'Fuel Share Petrol'].values[0]), 0.01)
    road_diesel_share = st.slider("Road: Diesel Share (2025)", 0.0, 1.0, float(road_df.loc[road_df['Year'] == 2025, 'Fuel Share Diesel'].values[0]), 0.01)
    road_ev_share = st.slider("Road: EV Share (2025)", 0.0, 1.0, float(road_df.loc[road_df['Year'] == 2025, 'Fuel Share EV'].values[0]), 0.01)
    # Normalize shares
    total_share = road_petrol_share + road_diesel_share + road_ev_share
    if total_share > 0:
        road_petrol_share /= total_share
        road_diesel_share /= total_share
        road_ev_share /= total_share
    # Apply to 2025 and beyond
    for y in years:
        if y >= 2025:
            road_df.loc[road_df['Year'] == y, 'Fuel Share Petrol'] = road_petrol_share
            road_df.loc[road_df['Year'] == y, 'Fuel Share Diesel'] = road_diesel_share
            road_df.loc[road_df['Year'] == y, 'Fuel Share EV'] = road_ev_share
    # (Repeat for rail and air if needed)
    # Add more controls for adoption rates, policy levers, etc.

if st.button("Run Scenario"):
    st.success("Scenario updated! All visualizations reflect the new parameters.")
    # In a full implementation, recalculate all outputs and update charts here.
    # (Future: Add scenario comparison, participatory features, etc.)

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
    # Calculate total emissions for 2022 and 2050 using backend functions
    total_emissions_2022 = (
        calculate_annual_emissions(road_df)[road_df["Year"] == 2022].values[0] +
        calculate_annual_emissions(rail_df)[rail_df["Year"] == 2022].values[0] +
        calculate_annual_emissions(air_df)[air_df["Year"] == 2022].values[0]
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
    calculate_annual_emissions(road_df)[road_df["Year"] == y].values[0] +
    calculate_annual_emissions(rail_df)[rail_df["Year"] == y].values[0] +
    calculate_annual_emissions(air_df)[air_df["Year"] == y].values[0]
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

# --- Dashboard Visualizations Section ---
st.header("Dashboard: Visualize Key Metrics")

# Line chart: Total CO2 emissions by year (all sectors)
st.subheader("Total CO₂ Emissions by Year (All Sectors)")
st.line_chart(emissions_df.set_index("Year")[["Total Emissions"]])

# Line chart: Total cost by year (all sectors)
st.subheader("Total Cost by Year (All Sectors)")
total_costs = [
    calculate_annual_cost(road_df)[road_df["Year"] == y].values[0] +
    calculate_annual_cost(rail_df)[rail_df["Year"] == y].values[0] +
    calculate_annual_cost(air_df)[air_df["Year"] == y].values[0]
    for y in years
]
costs_df = pd.DataFrame({"Year": years, "Total Cost": total_costs})
st.line_chart(costs_df.set_index("Year")[["Total Cost"]])

# Stacked bar chart: Miles traveled by sector (quick version with st.bar_chart)
st.subheader("Miles Traveled by Sector (Stacked Bar)")
miles_df = pd.DataFrame({
    "Year": years,
    "Road": [road_df.loc[road_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
    "Rail": [rail_df.loc[rail_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
    "Air": [air_df.loc[air_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
})
st.bar_chart(miles_df.set_index("Year"))

# (Optional) For more advanced/interactive charts, use plotly or matplotlib here.


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
