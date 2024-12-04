import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('wine_quality_model (1).pkl')

# Streamlit UI elements
st.title("Wine Quality Prediction")
st.write("Enter the features of the wine to predict its quality")

# Input form for user to enter wine features
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=2.0, value=0.0)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, value=1.9)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=0.2, value=0.076)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=100.0, value=34.0)
density = st.number_input('Density', min_value=0.990, max_value=1.003, value=0.9978)
pH = st.number_input('pH', min_value=2.0, max_value=4.0, value=3.51)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.number_input('Alcohol', min_value=8.0, max_value=15.0, value=9.4)

# Create input data in the correct format (as a tuple)
input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
              total_sulfur_dioxide, density, pH, sulphates, alcohol)

# Convert input data to numpy array and reshape it to 2D (model expects 2D input)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Create a button labeled 'Predict'
if st.button('Predict'):
    # Make prediction when the button is clicked
    prediction = model.predict(input_data_as_numpy_array)

    # Display the result
    if prediction[0] == 1:
        st.write("Prediction: Good Quality Wine 🍷")
    else:
        st.write("Prediction: Bad Quality Wine ❌")

# Footer
st.write("Made with ❤️ using Streamlit")
