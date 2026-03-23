# Maize Harvest Logistics DSS for Fleet Optimization

## Project Overview
This Decision Support System (DSS) helps farm managers determine the optimal number of external trucks to rent during harvest season. By predicting maize yield based on environmental factors, the system identifies the "Capacity Gap" between owned fleet and harvest volume.

## Team Members & Contributions
- **Deland Valdemar Ngong**: System Architecture, Data Cleaning, Model Development, Interface.
- **Nice Amoss**: Interface,Technical Documentation, Testing, and Data Acquisition.

## Technologies Used
- **Language**: Python 3.x
- **Libraries**: Pandas, Scikit-learn, Streamlit
- **Logic**: Random Forest Regression + Capacity Gap Analysis Algorithm

## System Architecture
1. **Data Component**: Preprocessed environmental dataset (Rainfall, Temp, Soil).
2. **Decision Engine**: Predictive model for yield estimation + Rule-based logic for fleet scaling.
3. **UI**: Interactive Dashboard for farm managers.
