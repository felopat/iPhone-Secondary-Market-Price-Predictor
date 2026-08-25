import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load latest scraped data
df = pd.read_csv("scraped_iphones.csv")

# Train pipeline on new dataset
X = df[[
    'model_generation', 
    'model_tier', 
    'storage_gb', 
    'condition_grade', 
    'is_unlocked', 
    'is_damaged_or_parts', 
    'device_age_years'
]]
y = df['price_usd']

categorical_cols = ['model_tier', 'condition_grade']
numeric_cols = ['model_generation', 'storage_gb', 'is_unlocked', 'is_damaged_or_parts', 'device_age_years']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X, y)

# Save updated dataset and model
df.to_pickle("iphone_price_prediction_500_rows.pkl")
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model updated successfully!")