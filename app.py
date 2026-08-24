import sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# NumPy 2.0+ pickle unpickling compatibility fix
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.numeric'] = np.core.numeric
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

st.set_page_config(page_title="iPhone Price Predictor", layout="centered")
st.title("📱 iPhone Market Price Predictor")
st.write("Fill in the specs below to estimate secondary market resale value.")

# Load dataset and train baseline pipeline model
@st.cache_resource
def load_and_train_model():
    df = pd.read_pickle("iphone_price_prediction_500_rows.pkl")
    
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
    return model_pipeline, df

pipeline, df = load_and_train_model()

st.divider()
st.subheader("Device Specifications")

# Center Input Layout using 2 Columns
col1, col2 = st.columns(2)

with col1:
    # Select Model Generation
    available_gens = sorted(df['model_generation'].unique())
    model_generation = st.selectbox(
        "Model Generation", 
        available_gens, 
        index=available_gens.index(13) if 13 in available_gens else 0
    )

    # Dynamic Filtering based on selected Generation
    df_gen = df[df['model_generation'] == model_generation]
    available_tiers = sorted(df_gen['model_tier'].unique())
    model_tier = st.selectbox("Model Tier", available_tiers)

    # Dynamic Filtering based on selected Tier
    df_gen_tier = df_gen[df_gen['model_tier'] == model_tier]
    available_storage = sorted(df_gen_tier['storage_gb'].unique())
    storage_gb = st.selectbox("Storage Size (GB)", available_storage)

with col2:
    # Automatically calculated device age
    device_age_years = int(df[df['model_generation'] == model_generation]['device_age_years'].iloc[0])
    st.number_input("Device Age (Years)", value=device_age_years, disabled=True)

    available_conditions = sorted(df['condition_grade'].unique())
    condition_grade = st.selectbox("Condition", available_conditions)

    st.write("---")
    is_unlocked = st.checkbox("Carrier Unlocked", value=True)
    is_damaged = st.checkbox("Damaged or For Parts", value=False)

st.divider()

# Prediction Button
if st.button("Estimate Market Price", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        'model_generation': model_generation,
        'model_tier': model_tier,
        'storage_gb': storage_gb,
        'condition_grade': condition_grade,
        'is_unlocked': int(is_unlocked),
        'is_damaged_or_parts': int(is_damaged),
        'device_age_years': device_age_years
    }])
    
    predicted_price = pipeline.predict(input_data)[0]
    st.success(f"### Estimated Price: **${predicted_price:,.2f}**")