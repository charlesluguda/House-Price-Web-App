import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

# Load the saved model
model = joblib.load('house_price_model.pkl')

# Load the California Housing dataset
housing = fetch_california_housing()

# Streamlit app
st.title('California Housing Price Prediction')

# Sidebar with user input
st.sidebar.header('User Input Features')

# Create text input boxes for user input features
# Median Income
median_income = st.sidebar.text_input('Median Income', value=f"{housing.data[:, 0].mean():.2f}")

# House Age
house_age = st.sidebar.text_input('House Age', value=f"{housing.data[:, 1].mean():.2f}")

# Average Rooms
avg_rooms = st.sidebar.text_input('Average Rooms', value=f"{housing.data[:, 2].mean():.2f}")

# Average Bedrooms
avg_bedrooms = st.sidebar.text_input('Average Bedrooms', value=f"{housing.data[:, 3].mean():.2f}")

# Population
population = st.sidebar.text_input('Population', value=f"{housing.data[:, 4].mean():.2f}")

# Average Occupation
avg_occupation = st.sidebar.text_input('Average Occupation', value=f"{housing.data[:, 5].mean():.2f}")

# Latitude
latitude = st.sidebar.text_input('Latitude', value=f"{housing.data[:, 6].mean():.2f}")

# Longitude
longitude = st.sidebar.text_input('Longitude', value=f"{housing.data[:, 7].mean():.2f}")

# Convert user input to DataFrame
user_input = {
    'MedInc': float(median_income),
    'HouseAge': float(house_age),
    'AveRooms': float(avg_rooms),
    'AveBedrms': float(avg_bedrooms),
    'Population': float(population),
    'AveOccup': float(avg_occupation),
    'Latitude': float(latitude),
    'Longitude': float(longitude)
}

input_df = pd.DataFrame([user_input])

# Predict the house price
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Predicted House Price')
st.write(f"${prediction[0] * 1000:.2f}")

# Optionally, you can add more visualizations or explanations based on your use case
