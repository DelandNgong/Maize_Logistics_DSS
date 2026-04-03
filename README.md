# Maize Harvest Logistics DSS for Fleet Optimization

## Project Overview
This Decision Support System (DSS) helps farm managers determine the optimal number of external trucks to rent during harvest season. By predicting maize yield based on environmental factors, the system identifies the "Capacity Gap" between owned fleet and harvest volume.

## Problem Statement
How many additional trucks must be rented to ensure the predicted maize yield is transported efficiently without exceeding the budget? 

## Technologies Used
Python 3.13: The core programming language for data processing and model logic.
Pandas: Used for high-performance data manipulation and cleaning of the 1.1M row dataset.
Scikit-Learn: Powering the Random Forest Regressor used for predictive yield modeling.
Streamlit: The framework used to build the interactive User Interface (UI) for the Farm Manager.
Joblib: For model serialization (saving/loading the trained .pkl "brain").
Git/GitHub: For version control and collaborative development.

## Team Members & Contributions
- **Deland Ngong**: System Architecture, Data Cleaning, Model Development, Interface.
- **Nice Amoss**: Interface,Technical Documentation, Testing, and Data Acquisition.

## Installation
**Clone repository** (git clone https://github.com/DelandNgong/Maize_Logistics_DSS)
**Instal Dependencies** (pip install streamlit pandas scikit-learn joblib)
**Generate Decision Model** (python src/train_model.py)
**Launch the Dashbaord** (streamlit run src/app.py)

## System Architecture
1. **Data Component**: Preprocessed environmental dataset (Rainfall, Temp, Soil).
2. **Decision Engine**: Predictive model for yield estimation + Rule-based logic for fleet scaling.
3. **UI**: Interactive Dashboard for farm managers.
