import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the cleaned data
DATA_PATH = 'data/maize_cleaned.csv'
MODEL_PATH = 'results/yield_model.pkl'

def train_yield_model():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Define Features (X) and Target (y)
    # Target is Yield_tons_per_hectare
    X = df.drop(columns=['Yield_tons_per_hectare'])
    y = df['Yield_tons_per_hectare']
    
    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize and Train Model
    print("Training the Yield Prediction Model...")

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Performance
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Trained! MAE: {mae:.2f}, R2 Score: {r2:.2f}")
    
    # 5. Save the Model for the UI
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_yield_model()