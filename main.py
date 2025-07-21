import streamlit as st
import pandas as pd
import numpy as np
import pulp
import requests
import os

# --- Data Initialization Functions ---
def make_default_df(sector):
    """Create default DataFrames for each transport sector with sample data."""
    years = list(range(2022, 2051))
    
    if sector == 'road':
        data = {
            'Year': years,
            'Miles Traveled': [1000000 + i*50000 for i in range(len(years))],
            'CO2 per Mile': [0.25 - i*0.005 for i in range(len(years))],
            'Cost per Mile': [0.12 - i*0.002 for i in range(len(years))],
            'Fuel Share Petrol': [0.7 - i*0.02 for i in range(len(years))],
            'Fuel Share Diesel': [0.2 - i*0.01 for i in range(len(years))],
            'Fuel Share EV': [0.1 + i*0.03 for i in range(len(years))]
        }
    elif sector == 'rail':
        data = {
            'Year': years,
            'Miles Traveled': [200000 + i*10000 for i in range(len(years))],
            'CO2 per Mile': [0.15 - i*0.003 for i in range(len(years))],
            'Cost per Mile': [0.08 - i*0.001 for i in range(len(years))],
            'Fuel Share Diesel': [0.8 - i*0.02 for i in range(len(years))],
            'Fuel Share Electric': [0.2 + i*0.02 for i in range(len(years))]
        }
    elif sector == 'air':
        data = {
            'Year': years,
            'Miles Traveled': [50000 + i*2000 for i in range(len(years))],
            'CO2 per Mile': [0.5 - i*0.008 for i in range(len(years))],
            'Cost per Mile': [0.25 - i*0.003 for i in range(len(years))],
            'Fuel Share Jet Fuel': [0.95 - i*0.01 for i in range(len(years))],
            'Fuel Share SAF': [0.05 + i*0.01 for i in range(len(years))]
        }
    else:
        raise ValueError(f"Unknown sector: {sector}")
    
    return pd.DataFrame(data)

# Initialize default data
years = list(range(2022, 2051))
road_df = make_default_df('road')
rail_df = make_default_df('rail')
air_df = make_default_df('air')

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

# --- Summary Output Section (Moved up for initialization) ---
"""
Summary Output Section:
Calculates and displays a table of total emissions by year, summing across all sub-sectors.
This provides a quick overview of the emissions trajectory for the current scenario.
"""
total_emissions = [
    calculate_annual_emissions(road_df)[road_df["Year"] == y].values[0] +
    calculate_annual_emissions(rail_df)[rail_df["Year"] == y].values[0] +
    calculate_annual_emissions(air_df)[air_df["Year"] == y].values[0]
    for y in years
]
emissions_df = pd.DataFrame({"Year": years, "Total Emissions": total_emissions})

# Calculate total costs for dashboard
total_costs = [
    calculate_annual_cost(road_df)[road_df["Year"] == y].values[0] +
    calculate_annual_cost(rail_df)[rail_df["Year"] == y].values[0] +
    calculate_annual_cost(air_df)[air_df["Year"] == y].values[0]
    for y in years
]
costs_df = pd.DataFrame({"Year": years, "Total Cost": total_costs})

# Calculate miles traveled for dashboard
miles_df = pd.DataFrame({
    "Year": years,
    "Road": [road_df.loc[road_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
    "Rail": [rail_df.loc[rail_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
    "Air": [air_df.loc[air_df["Year"] == y, "Miles Traveled"].values[0] for y in years],
})

# Add Net Zero logo to sidebar and header
st.logo(
    "https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png",
    link="https://www.tees.ac.uk/minisites/netzero/index.cfm",
    size="large"
)

# --- Sidebar Navigation ---
st.sidebar.title('🚦 Net Zero Transport Model')
st.sidebar.markdown('---')
section = st.sidebar.radio(
    'Jump to section:',
    [
        '🏠 Home',
        '📝 Data Input',
        '🔬 Scenario Modeling',
        '📈 Dashboard',
        '🤖 AI Optimization',
        '⚙️ Settings'
    ]
)

# --- Custom CSS for Color Scheme and Spacing ---
st.markdown(
    '''
    <style>
    body, [data-testid="stAppViewContainer"] {
        background-color: #f7fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        color: #2d572c;
    }
    .stButton>button {
        background-color: #6eb52f;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4e8c1a;
        color: #fff;
    }
    .stRadio>div>label {
        font-size: 1.1em;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# --- Section Routing ---
if section == '🏠 Home':
    st.title('🚦 Net Zero Transport Model')
    st.markdown('''
    Welcome to the Net Zero Transport Model Explorer! Use the sidebar to navigate between sections.
    ''')
    st.markdown('---')
    st.image("https://www.tees.ac.uk/minisites/netzero/images/netzero-logo.png", width=220)
    st.markdown('---')
    st.markdown('**Project by Teesside University**')

elif section == '📝 Data Input':
    st.header('📝 Data Input')
    st.markdown('---')
    st.markdown('Edit transport sector data below:')
    st.header('Editable Inputs by Sub-sector')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('🚗 Road')
        road_df = st.data_editor(road_df, num_rows="dynamic", key="road_editor")
    with col2:
        st.subheader('🚆 Rail')
        rail_df = st.data_editor(rail_df, num_rows="dynamic", key="rail_editor")
    with col3:
        st.subheader('✈️ Air')
        air_df = st.data_editor(air_df, num_rows="dynamic", key="air_editor")

elif section == '🔬 Scenario Modeling':
    st.header('🔬 Scenario Modeling')
    st.markdown('---')
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

elif section == '📈 Dashboard':
    st.header('📈 Dashboard: Visualize Key Metrics')
    st.markdown('---')
    st.subheader("Total CO₂ Emissions by Year (All Sectors)")
    st.line_chart(emissions_df.set_index("Year")[["Total Emissions"]])
    st.subheader("Total Cost by Year (All Sectors)")
    st.line_chart(costs_df.set_index("Year")[["Total Cost"]])
    st.subheader("Miles Traveled by Sector (Stacked Bar)")
    st.bar_chart(miles_df.set_index("Year"))

elif section == '🤖 AI Optimization':
    st.header('🤖 AI-Assisted Scenario/Objective Input')
    st.markdown('---')
    st.markdown("After entering your scenario, click the **'Generate Optimization Parameters with AI'** button below.")
    user_prompt = st.text_area(
        "Describe your optimization goal or scenario (natural language):",
        placeholder="e.g., Minimize emissions while keeping costs under £1M and EV share above 30% by 2030."
    )

    def call_gemini_and_parse(prompt, api_key=None):
        """
        Calls Google Gemini API with the user's prompt and parses the response into a params dict.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")  # Or set your key directly here

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            gemini_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return parse_llm_response(gemini_text)
        else:
            st.error(f"Gemini API error: {response.status_code} {response.text}")
            return None

    # Example parser for LLM response (to be customized for your prompt/response format)
    def parse_llm_response(llm_text):
        """
        Parse the LLM's text response into a params dictionary.
        This is a placeholder; implement robust parsing for your use case.
        """
        # Example: Use regex, json.loads, or custom logic
        # For now, return a static dict
        return {
            'objective': 'cost',
            'row7_constraint': True,
            'max_change_per_year': None,
            'block_constraints': True,
            'equality_constraints': True,
        }

    # --- Optimization Model (from Excel Sheet) ---
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

        # --- AI/User parameter input section ---
        params = {
            'objective': 'cost',  # 'cost', 'emissions', or 'weighted'
            'row7_constraint': True,
            'max_change_per_year': None,  # e.g., 10000 or None
            'block_constraints': True,
            'equality_constraints': True,
            # Add more parameters as needed
        }
        if st.button("Generate Optimization Parameters with AI"):
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                with st.spinner("Contacting Gemini AI..."):
                    ai_params = call_gemini_and_parse(user_prompt, api_key)
                if ai_params:
                    params = ai_params
                    st.write("**AI-generated optimization parameters:**", params)
                else:
                    st.warning("Failed to generate parameters from Gemini.")
            except:
                st.warning("Gemini API key not configured. Using default parameters.")

        add_objective(model, model_df, variables, params)
        add_constraints(model, model_df, variables, params)

        # Solve the model
        model.solve()

        st.write("**Optimization Status:**", pulp.LpStatus[model.status])
        st.write("**Objective value:**", pulp.value(model.objective))
        # Optionally, display decision variables
        # for var in model.variables():
        #     st.write(var.name, "=", var.varValue)
    else:
        st.info("Upload the REHIP Model Excel file to run the optimization model.")

elif section == '⚙️ Settings':
    st.header('⚙️ Settings & Info')
    st.markdown('---')
    st.markdown('Customize your experience or view app info here.')
    # ... (future settings, about, etc.) ...

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
# Recalculate emissions with current data
total_emissions = [
    calculate_annual_emissions(road_df)[road_df["Year"] == y].values[0] +
    calculate_annual_emissions(rail_df)[rail_df["Year"] == y].values[0] +
    calculate_annual_emissions(air_df)[air_df["Year"] == y].values[0]
    for y in years
]
emissions_df = pd.DataFrame({"Year": years, "Total Emissions": total_emissions})
st.dataframe(emissions_df)

# --- Modular Optimization Logic Section ---
def add_objective(model, model_df, variables, params):
    """
    Adds the objective function to the model based on params['objective'].
    Extend this function to support cost, emissions, or weighted objectives.
    """
    if params.get('objective', 'cost') == 'cost':
        objective_value = model_df.iloc[397, 1]
        model += objective_value, "Total_Cost_Objective"
    elif params['objective'] == 'emissions':
        # Example: model += emissions_expr, "Total_Emissions_Objective"
        pass
    # Add more options as needed

def add_constraints(model, model_df, variables, params):
    """
    Adds constraints to the model based on params.
    Extend this function to support toggling constraints, max change per year, etc.
    """
    # Row 7 constraint
    if params.get('row7_constraint', True):
        row7_values = model_df.iloc[6, 1:30].fillna(0).values
        constraint_rhs = model_df.iloc[391, 1]
        model += pulp.lpSum(row7_values) <= constraint_rhs, "Row7_Constraint"
    # Max change per year constraint (example, can be expanded)
    if params.get('max_change_per_year', None) is not None:
        max_change = params['max_change_per_year']
        for i in range(68, 101):  # Example: years 68-100
            for j in range(2, 31):
                model += variables[(i, j)] - variables[(i-1, j)] <= max_change, f"MaxIncrease_{i}_{j}"
                model += variables[(i-1, j)] - variables[(i, j)] <= max_change, f"MaxDecrease_{i}_{j}"
    # Block constraints (optional, can be toggled)
    if params.get('block_constraints', True):
        row_pairs = [(181, 439), (231, 446), (281, 453), (331, 460), (131, 432)]
        for idx, (lhs_start, rhs_start) in enumerate(row_pairs):
            lhs_block = model_df.iloc[lhs_start:lhs_start+6, 2:31].fillna(0).values
            rhs_block = model_df.iloc[rhs_start:rhs_start+6, 2:31].fillna(0).values
            for i in range(6):
                for j in range(29):
                    model += variables[(i+lhs_start, j+2)] <= rhs_block[i][j], f"Block_{idx}_r{i}_c{j}"
    # Equality constraints
    if params.get('equality_constraints', True):
        for row in range(109, 114):
            values = model_df.iloc[row, 2:31].fillna(0).values
            model += pulp.lpSum(values) == 100, f"Equality_Row_{row+1}"
    # Add more constraints as needed

# --- AI/LLM Natural Language Scenario Input Section ---
st.header("AI-Assisted Scenario/Objective Input")
st.markdown("After entering your scenario, click the **'Generate Optimization Parameters with AI'** button below.")
user_prompt = st.text_area(
    "Describe your optimization goal or scenario (natural language):",
    placeholder="e.g., Minimize emissions while keeping costs under £1M and EV share above 30% by 2030."
)

def call_gemini_and_parse(prompt, api_key=None):
    """
    Calls Google Gemini API with the user's prompt and parses the response into a params dict.
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")  # Or set your key directly here

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        gemini_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return parse_llm_response(gemini_text)
    else:
        st.error(f"Gemini API error: {response.status_code} {response.text}")
        return None

# Example parser for LLM response (to be customized for your prompt/response format)
def parse_llm_response(llm_text):
    """
    Parse the LLM's text response into a params dictionary.
    This is a placeholder; implement robust parsing for your use case.
    """
    # Example: Use regex, json.loads, or custom logic
    # For now, return a static dict
    return {
        'objective': 'cost',
        'row7_constraint': True,
        'max_change_per_year': None,
        'block_constraints': True,
        'equality_constraints': True,
    }

# --- Optimization Model (from Excel Sheet) ---
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

    # --- AI/User parameter input section ---
    params = {
        'objective': 'cost',  # 'cost', 'emissions', or 'weighted'
        'row7_constraint': True,
        'max_change_per_year': None,  # e.g., 10000 or None
        'block_constraints': True,
        'equality_constraints': True,
        # Add more parameters as needed
    }
    if st.button("Generate Optimization Parameters with AI"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            with st.spinner("Contacting Gemini AI..."):
                ai_params = call_gemini_and_parse(user_prompt, api_key)
            if ai_params:
                params = ai_params
                st.write("**AI-generated optimization parameters:**", params)
            else:
                st.warning("Failed to generate parameters from Gemini.")
        except:
            st.warning("Gemini API key not configured. Using default parameters.")

    add_objective(model, model_df, variables, params)
    add_constraints(model, model_df, variables, params)

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
# Recalculate costs with current data
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
# Recalculate miles with current data
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
