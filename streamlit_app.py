
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your trained model

st.title('Car Price Prediction App')

st.write('This app predicts the selling price of a car based on its attributes.')

# --- Model Loading ---
# Load your trained model here. Make sure the path is correct.
try:
    model = joblib.load('best_gb_model.pkl') # Assuming you saved your best Gradient Boosting model
except FileNotFoundError:
    st.error("Model file 'best_gb_model.pkl' not found. Please ensure the model is saved in the same directory.")
    st.stop() # Stop the app if the model is not found

# --- Input Fields ---
# Add input fields for the features your model was trained on.
# Refer to the features used in your X_final DataFrame:
# ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol', 'transmission_type_Manual']

# Example Input fields (customize based on your features and their types):
vehicle_age = st.slider('Vehicle Age', 0, 30, 5)
km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=4000000, value=50000)
mileage = st.number_input('Mileage (kmpl)', min_value=0.0, max_value=40.0, value=15.0, step=0.1)
engine = st.number_input('Engine (CC)', min_value=500, max_value=7000, value=1200)
max_power = st.number_input('Max Power (BHP)', min_value=0.0, max_value=700.0, value=80.0, step=0.1)
seats = st.selectbox('Number of Seats', [4, 5, 6, 7, 8, 9])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])


# --- Prediction ---
# Create a button to trigger prediction
if st.button('Predict Selling Price'):
    # Prepare the input data as a pandas DataFrame, matching the structure used during training
    input_data = pd.DataFrame([[vehicle_age, km_driven, mileage, engine, max_power, seats, seller_type, fuel_type, transmission_type]],
                              columns=['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'seller_type', 'fuel_type', 'transmission_type'])

    # Preprocess the input data (e.g., one-hot encoding, scaling) - MUST EXACTLY MATCH TRAINING PREPROCESSING
    # Example (assuming one-hot encoding for categorical features):
    input_data_processed = pd.get_dummies(input_data, columns=['seller_type', 'fuel_type', 'transmission_type'], drop_first=True)

    # Ensure all columns present during training are also present in the input data (with 0 if not present)
    # This step is crucial for consistent feature order and presence
    # Get the list of columns your model was trained on from X_train or X_final
    training_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 'fuel_type_Petrol', 'transmission_type_Manual'] # Replace with actual training columns from your notebook

    for col in training_columns:
        if col not in input_data_processed.columns:
            input_data_processed[col] = 0
    input_data_processed = input_data_processed[training_columns] # Ensure column order is the same as training data

    # Make prediction
    try:
        prediction = model.predict(input_data_processed)
        st.success(f'Predicted Selling Price: {prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


