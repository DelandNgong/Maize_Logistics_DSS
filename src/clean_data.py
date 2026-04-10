import pandas as pd
import os

# 1. Define Paths (This ensures files go to the right folders)
RAW_DATA = 'data/dataset.csv'
CLEAN_DATA = 'data/maize_cleaned.csv'

def clean_data():
    # Load the 90MB file
    print("Reading raw data...")
    df = pd.read_csv(RAW_DATA)

    # 2. Filter for Maize
    maize_df = df[df['Crop'] == 'Maize'].copy()

    # 3. Feature Selection
    # Removing Region and Days_to_Harvest as they aren't primary yield predictors.
    # We drop 'Crop' because every row is now Maize.
    maize_df.drop(columns=['Region', 'Crop', 'Days_to_Harvest'], inplace=True, errors='ignore')

    # 4. Encoding Categorical Data
    # Convert Booleans to 1/0 for the model to process numerically.
    maize_df['Fertilizer_Used'] = maize_df['Fertilizer_Used'].astype(int)
    maize_df['Irrigation_Used'] = maize_df['Irrigation_Used'].astype(int)

    # One-Hot Encoding for Soil and Weather (creates separate columns for each type)
    maize_df = pd.get_dummies(maize_df, columns=['Soil_Type', 'Weather_Condition'])

    # 5. Sampling for GitHub 
    # We take 15,000 rows, under GitHub's 100MB limit.
    final_df = maize_df.sample(n=15000, random_state=42)

    # 6. Save directly to the data folder
    final_df.to_csv(CLEAN_DATA, index=False)
    print(f"Success! {len(final_df)} rows saved to {CLEAN_DATA}")

if __name__ == "__main__":
    clean_data()