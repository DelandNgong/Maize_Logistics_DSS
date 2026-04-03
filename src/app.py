import streamlit as st
import pandas as pd
import joblib
import math

# 1. System Configuration & Setup
st.set_page_config(page_title="Maize Logistics DSS", layout="wide")

# Load the trained model (Ensure this file exists in your results/ folder)
MODEL_PATH = 'results/maize_yield_model.pkl' 
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}. Please run src/train_model.py first.")

# --- 2. User Interface Header ---
st.title("🌽 Maize Harvest Logistics DSS")
st.markdown("### Decision Support for Fleet Optimization")
st.info("This system predicts maize yield and recommends truck rentals based on a Capacity Gap Analysis.")

# --- 3. Sidebar: Environmental Inputs (Data Component) ---
st.sidebar.header("1. Farm & Environmental Data")

# Numeric inputs for the predictive model
hectares = st.sidebar.number_input("Total Farm Size (Hectares)", min_value=1.0, value=10.0, step=1.0)
rainfall = st.sidebar.slider("Seasonal Rainfall (mm)", 100, 1500, 500)
temp = st.sidebar.slider("Avg Temperature (°C)", 10, 45, 25)

# Boolean inputs for farming practices
fertilizer = st.sidebar.selectbox("Fertilizer Used?", [True, False])
irrigation = st.sidebar.selectbox("Irrigation Used?", [True, False])

# Categorical selections for Soil and Weather
soil_type = st.sidebar.selectbox("Soil Type", ["Clay", "Loam", "Sandy", "Silt", "Chalky", "Peaty"])
weather = st.sidebar.selectbox("Weather Condition", ["Cloudy", "Rainy", "Sunny"])

# --- 4. Main Page: Logistic Inputs (Decision Maker Variables) ---
st.header("2. Logistics & Fleet Capacity")
col1, col2 = st.columns(2)

with col1:
    owned_trucks = st.number_input("Number of Owned Trucks", min_value=0, value=5, step=1)
with col2:
    truck_capacity = st.number_input("Capacity per Truck (Tons)", min_value=1, value=10, step=1)

# --- 5. Decision Engine Logic ---
if st.button("Generate Logistic Recommendation"):
    # Create the feature vector for the model
    # Note: Column names must match the training set (maize_cleaned.csv) exactly
    raw_input = {
        'Rainfall_mm': [float(rainfall)],
        'Temperature_Celsius': [float(temp)],
        'Fertilizer_Used': [int(fertilizer)],
        'Irrigation_Used': [int(irrigation)],
        'Soil_Type_Chalky': [1 if soil_type == "Chalky" else 0],
        'Soil_Type_Clay': [1 if soil_type == "Clay" else 0],
        'Soil_Type_Loam': [1 if soil_type == "Loam" else 0],
        'Soil_Type_Peaty': [1 if soil_type == "Peaty" else 0],
        'Soil_Type_Sandy': [1 if soil_type == "Sandy" else 0],
        'Soil_Type_Silt': [1 if soil_type == "Silt" else 0],
        'Weather_Condition_Cloudy': [1 if weather == "Cloudy" else 0],
        'Weather_Condition_Rainy': [1 if weather == "Rainy" else 0],
        'Weather_Condition_Sunny': [1 if weather == "Sunny" else 0]
    }
    
    input_df = pd.DataFrame(raw_input)

    try:
        # Step A: Synchronize feature order with the model's training state
        input_df = input_df[model.feature_names_in_]
        
        # Step B: Yield Prediction (The Predictive Model component)
        yield_per_hectare = model.predict(input_df)[0]
        total_predicted_yield = yield_per_hectare * hectares
        
        # Step C: Capacity Gap Analysis (The Optimization Logic)
        total_owned_capacity = owned_trucks * truck_capacity
        capacity_gap = total_predicted_yield - total_owned_capacity
        
        # Recommendation: Rent additional trucks if yield exceeds capacity
        trucks_to_rent = math.ceil(capacity_gap / truck_capacity) if capacity_gap > 0 else 0

        # --- 6. Results Display (System Output) ---
        st.divider()
        st.subheader("Decision Support Summary")
        
        # Display key metrics in columns for scannability
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Yield", f"{total_predicted_yield:.2f} Tons")
        res2.metric("Owned Capacity", f"{total_owned_capacity} Tons")
        
        if trucks_to_rent > 0:
            res3.error(f"Action: Rent {trucks_to_rent} Trucks")
            st.warning(f"LOGISTICS ALERT: Current fleet is {capacity_gap:.2f} tons short of demand.")
        else:
            res3.success("Action: Use Owned Fleet")
            st.balloons() # Visual indicator of success
            st.info("LOGISTICS OK: Your current fleet can efficiently transport the harvest.")
            
    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.write("Ensure your training script and app use consistent feature names.")