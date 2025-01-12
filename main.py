import streamlit as st
import pandas as pd
import pickle

# Load the dataset and the model
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Set up the Streamlit app
st.title("House Price Prediction App")
st.write("Provide the following details to predict the house price:")

# Input fields for user
bedrooms = st.selectbox("Number of Bedrooms", sorted(data['beds'].unique()))
bathrooms = st.selectbox("Number of Bathrooms", sorted(data['baths'].unique()))
size = st.number_input("Size (in square feet)", min_value=float(data['size'].min()), max_value=float(data['size'].max()))
zip_code = st.selectbox("Zip Code", sorted(data['zip_code'].unique()))

# Create a DataFrame with input data
input_data = pd.DataFrame([[bedrooms, bathrooms, size, zip_code]],
                          columns=['beds', 'baths', 'size', 'zip_code'])

# Handle unknown categories in the input data
for column in input_data.columns:
    unknown_categories = set(input_data[column]) - set(data[column].unique())
    if unknown_categories:
        st.warning(f"Unknown categories detected in {column}: {unknown_categories}. Using default value.")
        input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

# Prediction
if st.button("Predict"):
    # Make predictions
    prediction = pipe.predict(input_data)[0]
    st.success(f"The predicted house price is: ₹{prediction:,.2f}")
