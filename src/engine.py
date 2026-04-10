import joblib
import pandas as pd

# Load the trained model
MODEL_PATH = 'results/yield_model.pkl'

def calculate_logistics(input_data, owned_trucks, truck_capacity):
    """
    Logic: Number of Rental Trucks = (Total Yield - Owned Capacity) / Capacity per truck
    """
    # 1. Load Model
    model = joblib.load(MODEL_PATH)
    
    # 2. Predict Yield (Tons per Hectare)
    yield_per_hectare = model.predict(input_data)[0]
    
    # 3. Capacity Gap Analysis
    total_yield = yield_per_hectare * input_data['Hectares'].iloc[0] # Assuming user inputs Hectares
    owned_capacity = owned_trucks * truck_capacity
    
    surplus = max(0, total_yield - owned_capacity)
    trucks_to_rent = int(pd.np.ceil(surplus / truck_capacity)) if surplus > 0 else 0
    
    return {
        "predicted_yield": round(total_yield, 2),
        "owned_capacity": owned_capacity,
        "rental_recommendation": trucks_to_rent
    }