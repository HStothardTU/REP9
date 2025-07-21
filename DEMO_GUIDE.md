# 🚦 Net Zero Transport Model - Demo Guide

## Quick Start for Demo

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python run_demo.py
```
Or directly:
```bash
streamlit run main.py
```

The app will open at: http://localhost:8501

## Demo Walkthrough

### 🏠 Home Section
- **Show**: Welcome page with Net Zero branding
- **Highlight**: Professional appearance and Teesside University branding

### 📝 Data Input Section
- **Show**: Editable tables for Road, Rail, and Air sectors
- **Demo**: 
  - Edit some values in the tables
  - Show how data updates in real-time
  - Explain the different fuel types and their trends

### 🔬 Scenario Modeling Section
- **Show**: Interactive sliders for fuel share adjustments
- **Demo**:
  - Adjust the Road EV share slider
  - Click "Run Scenario" to see updates
  - Explain how this affects emissions and costs

### 📈 Dashboard Section
- **Show**: Three key visualizations
- **Demo**:
  - **CO₂ Emissions Chart**: Shows declining trend
  - **Cost Chart**: Shows cost projections
  - **Miles Traveled Chart**: Shows sector breakdown
- **Highlight**: Real-time updates when data changes

### 🤖 AI Optimization Section
- **Show**: Natural language input for optimization
- **Demo**:
  - Type: "Minimize emissions while keeping costs under £1M"
  - Click "Generate Optimization Parameters with AI"
  - Show how AI interprets the request
- **Note**: Requires Gemini API key for full functionality

### Scenario Management Features
- **Show**: Save/Load scenarios
- **Demo**:
  - Save current scenario as "Demo Scenario"
  - Make some changes
  - Load the saved scenario to restore original data

### Backcasting Feature
- **Show**: Goal-setting functionality
- **Demo**:
  - Enable backcasting
  - Set target emissions for 2050 (e.g., 0 for net zero)
  - Show required annual percentage change

## Key Features to Highlight

### ✅ Working Features
- ✅ Interactive data editing
- ✅ Real-time calculations
- ✅ Scenario management
- ✅ Multiple visualizations
- ✅ Professional UI/UX
- ✅ Backcasting calculations
- ✅ Optimization framework (with Excel upload)

### 🔧 Demo-Ready Features
- ✅ Sample data for all sectors
- ✅ Error handling for missing API keys
- ✅ Responsive design
- ✅ Clear navigation

### 📊 Sample Data Included
- **Road**: 1M+ miles, declining emissions, EV adoption
- **Rail**: 200K miles, electrification trend
- **Air**: 50K miles, SAF adoption

## Troubleshooting

### If the app doesn't start:
1. Check all dependencies are installed: `pip install -r requirements.txt`
2. Ensure you're in the correct directory
3. Try: `streamlit run main.py --server.port=8501`

### If charts don't update:
1. Make sure to click "Run Scenario" after making changes
2. Check that data tables have valid numbers

### If AI features don't work:
- This is expected without a Gemini API key
- The app will show a warning and use default parameters

## Demo Tips

1. **Start with Home** - Show the professional branding
2. **Quick Data Edit** - Make a small change to show interactivity
3. **Dashboard** - Show the visualizations working
4. **Scenario Save/Load** - Demonstrate persistence
5. **Backcasting** - Show the goal-setting feature
6. **AI Section** - Mention the optimization capabilities

## Technical Notes

- Built with Streamlit for rapid development
- Uses PuLP for optimization (when Excel file provided)
- Sample data covers 2022-2050
- All calculations update in real-time
- Professional Net Zero branding throughout 