# Net Zero Transport Model

An interactive, evidence-based decision support tool for sustainable fuel deployment in Teesside’s transport sector.

## Project Overview
This Streamlit app enables users to analyze, model, and compare future fuel scenarios for road, rail, and air transport. It supports participatory scenario building, backcasting, and transparent calculation of emissions, costs, and fuel use. The tool is designed for researchers, policymakers, and stakeholders to collaboratively explore decarbonization pathways and policy options.

## Key Features
- **Editable input tables** for Road, Rail, and Air sub-sectors (miles, fuel shares, costs, emissions, etc.)
- **Scenario builder**: Save, load, and compare multiple scenarios
- **Backcasting**: Set end goals (e.g., 2050 emissions target) and see required annual changes
- **Summary outputs**: Year-by-year emissions table and scenario results
- **Branding**: Custom logo and color scheme for Teesside Net Zero

## Setup Instructions
1. **Clone the repository**
   ```sh
   git clone https://github.com/HStothardTU/REP9.git
   cd REP9
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```sh
   streamlit run main.py
   ```

## Usage Guide
- **Editable Inputs**: Adjust miles traveled, fuel shares, costs, and emission factors for each sub-sector and year.
- **Scenario Management**: Save the current scenario with a name, load any saved scenario, and compare results.
- **Backcasting**: Enable backcasting, set a 2050 emissions target, and see the required annual % change to meet the goal.
- **Summary Table**: View total emissions by year for the current scenario.
- **Branding**: The app uses the Net Zero logo and color scheme for a professional, region-specific look.

## Project Structure
- `main.py` — Main Streamlit app with all logic and UI
- `assets/netzero-logo.svg` — Local logo for branding
- `.streamlit/config.toml` — Custom theme configuration
- `requirements.txt` — Python dependencies

## Contact
For questions, feedback, or collaboration, contact:
- Project Lead: Hannah Stothard
- Email: [your-email@domain.com]
- [Teesside University Net Zero](https://www.tees.ac.uk/minisites/netzero/index.cfm)

---

*This tool is part of a research project to provide an evidence-based roadmap for sustainable fuel deployment in Teesside’s transportation sector. It is open for collaboration and further development.* 