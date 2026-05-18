import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and scaler
try:
    linear_regressor = pickle.load(open('linear_regression_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'linear_regression_model.pkl' and 'scaler.pkl' exist in the root directory.")
    st.stop()

# Get the maximum year from the original dataset (replace with actual max year if known)
# For deployment, it's best to save this value or hardcode it if it's constant
# Let's assume you have a way to get the max_year (e.g., from a saved config or hardcoded)
# For now, I'll use the max_year from the training data context, but ideally, this would be a saved parameter.
# If you don't save max_year, you might need to adjust Car_age calculation in app.py or save a more complete preprocessing pipeline.
# For this example, let's hardcode a reasonable max_year or retrieve it from a placeholder.

# Placeholder for max_year_in_dataset - in a real scenario, you'd save this with your model assets
# For the purpose of this example, we'll use 2018 as derived earlier in the notebook.
max_year_in_dataset = 2018 # This should come from your saved training context

st.title('Car Price Prediction App')
st.write('Enter car details to predict its selling price in Lakhs.')

# Input features from the user
present_price = st.number_input('Present Price (in Lakhs)', min_value=0.5, max_value=100.0, value=5.0, step=0.1)
kms_driven = st.number_input('Kilometers Driven', min_value=100, max_value=500000, value=50000, step=1000)
owner = st.selectbox('Number of Owners', options=[0, 1, 2, 3])
year = st.slider('Manufacturing Year', min_value=2000, max_value=max_year_in_dataset, value=2015)

fuel_type = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', options=['Dealer', 'Individual'])
transmission = st.selectbox('Transmission Type', options=['Manual', 'Automatic'])

# Calculate Car_age
car_age = max_year_in_dataset - year

if st.button('Predict Selling Price'):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame([[present_price, kms_driven, owner, car_age,
                                (1 if fuel_type == 'Diesel' else 0),
                                (1 if fuel_type == 'Petrol' else 0),
                                (1 if seller_type == 'Individual' else 0),
                                (1 if transmission == 'Manual' else 0)]],
                              columns=['Present_Price', 'Kms_Driven', 'Owner', 'Car_age',
                                       'Fuel_Type_Diesel', 'Fuel_Type_Petrol', # Note: CNG is the reference, so it's 0 for both
                                       'Seller_Type_Individual', 'Transmission_Manual'])

    # Handle Fuel_Type_CNG (it should be 0 if Diesel/Petrol are 1, else 1 if CNG is selected)
    # The original get_dummies uses drop_first=True, so only Diesel and Petrol columns exist.
    # If fuel_type is CNG, both Fuel_Type_Diesel and Fuel_Type_Petrol should be 0.
    if fuel_type == 'CNG':
        input_data['Fuel_Type_Diesel'] = 0
        input_data['Fuel_Type_Petrol'] = 0

    # Ensure columns are in the same order as during training
    # (This assumes the order from the original X_train was consistent after get_dummies)
    # It's crucial that the column order and names match exactly
    # based on the X_train used for fitting the scaler and model.
    # From the context, the columns were: 'Present_Price', 'Kms_Driven', 'Owner', 'Car_age', 
    # 'Fuel_Type_Diesel', 'Fuel_Type_Petrol' (if both existed for a non-binary fuel), 
    # 'Seller_Type_Individual', 'Transmission_Manual'
    # Let's verify the exact columns of X_train from kernel state.
    # Based on kernel state X: 'Present_Price', 'Kms_Driven', 'Owner', 'Car_age', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual'
    # The actual order from X (from kernel state) has Fuel_Type_Diesel, Fuel_Type_Petrol (assuming presence for Petrol)
    # However, the previous get_dummies used `drop_first=True`, so only Fuel_Type_Diesel and Fuel_Type_Petrol are likely.
    # Let's adjust based on the structure of df_encoded in the kernel state.
    # df_encoded columns: 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner', 'Car_age', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual'
    # So, X columns are all of these except 'Selling_Price'.

    # Re-order columns to match training data (important for consistent predictions)
    training_columns = ['Present_Price', 'Kms_Driven', 'Owner', 'Car_age', 
                        'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 
                        'Seller_Type_Individual', 'Transmission_Manual']
    
    # Ensure all columns expected by the model are present, even if their value is 0
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[training_columns]

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = linear_regressor.predict(scaled_input_data)[0]

    st.success(f'Predicted Selling Price: {prediction:.2f} Lakhs')
