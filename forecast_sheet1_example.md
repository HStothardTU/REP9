# Sheet1 Forecast Structure and Model Logic (Bus Example)

## Sheet Structure
- **Columns:**
  - 'Historical', 'Forecast', and subsequent columns for each year (e.g., 2022, 2023, ..., 2040)
- **Rows:**
  - Year (header row)
  - Bus distance [Mmiles]: Total annual bus miles
  - Diesel bus [%], HEV bus [%], Other fuel bus [%], Petrol bus [%], EV bus [%]: Share of each fuel type in the bus fleet (percent, sum to 100%)
  - Sum [%]: Check row (should always be 100%)
  - Miles by fuel [Mmiles/year]:
    - Diesel, Petrol, HEV, Other, EV: Annual miles by each fuel type (calculated)
    - Sum: Check row (should match total bus miles)
  - Emission factor [kgCO2e/mile]:
    - Diesel, Petrol, HEV, Other: Emission factor for each fuel type
  - (Other rows may follow for cost, etc.)

## Calculation Logic

### 1. **Fuel Shares**
- For each year, the share of each fuel type (Diesel, HEV, Other, Petrol, EV) is specified as a percentage.
- The sum of all fuel shares for a year should be 100% (see 'Sum [%]').

### 2. **Miles by Fuel**
- For each year and fuel type:
  - **Miles by fuel = Total bus miles × Fuel share [%] / 100**
  - Example: Diesel miles in 2023 = Bus distance in 2023 × Diesel bus [%] in 2023 / 100
- The sum of all 'Miles by fuel' for a year should match the total bus miles for that year.

### 3. **Emissions Calculation**
- For each year and fuel type:
  - **Emissions = Miles by fuel × Emission factor [kgCO2e/mile]**
  - Example: Diesel emissions in 2023 = Diesel miles in 2023 × Diesel emission factor
- Total bus emissions for a year = Sum of emissions from all fuel types.

### 4. **Forecasting**
- The 'Forecast' columns allow for projecting changes in fuel shares, total miles, and emission factors over time.
- By adjusting fuel shares (e.g., increasing EV share, decreasing diesel), the model projects the impact on miles by fuel and total emissions.

## Example Calculation (2023, Diesel):
- Bus distance [Mmiles]: 22.53
- Diesel bus [%]: 87.8
- Diesel miles = 22.53 × 87.8 / 100 = 19.78 Mmiles
- Diesel emission factor: 1.0074 kgCO2e/mile
- Diesel emissions = 19.78 × 1.0074 = 19.92 million kgCO2e

## Notes
- The same logic applies for other vehicle types and fuels if present in the sheet.
- This structure enables scenario analysis by adjusting fuel shares and emission factors for future years. 