import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Step 1: Generate Fake Data for 10 African Countries
np.random.seed(42)
countries = [
    "Nigeria", "Ethiopia", "Egypt", "Democratic Republic of the Congo", "Tanzania",
    "South Africa", "Kenya", "Uganda", "Sudan", "Algeria", "Rwanda"
]

# Create fake data for each country
data_dict = {}
for country in countries:
    months = np.arange(1, 25)  # Simulate 24 months
    total_cases = np.cumsum(np.random.randint(10, 100, size=24))  # Cumulative sum to simulate total cases
    previous_day_total_cases = np.roll(total_cases, 1)  # Previous day's total cases
    previous_day_total_cases[0] = 0  # For the first month, there's no previous day data
    
    # Create DataFrame for each country
    data_dict[country] = pd.DataFrame({
        'month': months,
        'total_cases': total_cases,
        'previous_day_total_cases': previous_day_total_cases
    })
    
    # Add rolling average
    data_dict[country]['rolling_avg'] = data_dict[country]['previous_day_total_cases'].rolling(window=3).mean()
    data_dict[country]['rolling_avg'] = data_dict[country]['rolling_avg'].fillna(data_dict[country]['previous_day_total_cases'])

# Step 2: Prepare Data for Model Training (for each country)
features = ['month', 'previous_day_total_cases', 'rolling_avg']
target = 'total_cases'

# Store the trained models for each country
models = {}
scalers = {}

# Train a model for each country
for country, data in data_dict.items():
    X = data[features]
    y = data[target]

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scalers[country] = scaler

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train an XGBoost Model
    model = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=1.0, learning_rate=0.1, max_depth=5,
                             n_estimators=100, reg_lambda=10, subsample=0.8)
    model.fit(X_train, y_train)

    # Store the trained model
    models[country] = model

# Step 3: Setup Streamlit UI for Country Selection
st.title("COVID-19 Total Cases Prediction by Country")

# List of countries to choose from
country_choices = list(countries)

# Sidebar with country selection
selected_country = st.sidebar.selectbox("Select a Country", country_choices)

# Display the model's performance (Mean Squared Error, R-squared)
selected_data = data_dict[selected_country]
X = selected_data[features]
y = selected_data[target]

# Normalize the data
scaler = scalers[selected_country]
X_scaled = scaler.transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Get the trained model for the selected country
model = models[selected_country]

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Evaluation Metrics
st.write(f"Mean Squared Error for {selected_country}: {mse:.2f}")
st.write(f"R-squared for {selected_country}: {r2:.2f}")

# Step 4: Predict Future Data for the Selected Country
future_months = np.array([25, 26, 27])  # Predict for the next 3 months
last_day_total_cases = selected_data['total_cases'].iloc[-1]  # Get the last month's total cases

# Use the last rolling average for future data (same as the latest value)
last_rolling_avg = selected_data['rolling_avg'].iloc[-1]

# Create future data DataFrame for prediction
future_data = pd.DataFrame({
    'month': future_months,
    'previous_day_total_cases': [last_day_total_cases] * 3,
    'rolling_avg': [last_rolling_avg] * 3  # Use the last rolling average value for simplicity
})

# Normalize the future data
future_data_scaled = scaler.transform(future_data)

# Predict future cases
predicted_cases = model.predict(future_data_scaled)
future_data['predicted_total_cases'] = predicted_cases

# Display the forecast for the selected country
st.write(f"Forecast for {selected_country} (Next 3 months):")
st.write(future_data)

# Step 5: Plot the Actual vs Predicted Data for the Selected Country
plt.figure(figsize=(10, 6))
plt.plot(selected_data['month'], selected_data['total_cases'], label='Actual Total Cases', color='blue')
plt.plot(future_data['month'], future_data['predicted_total_cases'], label='Predicted Total Cases', color='red', linestyle='--')
plt.title(f'Actual vs Predicted Total Cases for {selected_country}')
plt.xlabel('Month')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)