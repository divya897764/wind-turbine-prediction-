import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
print("📂 Current working directory:", os.getcwd())


# Load dataset
df = pd.read_csv("wind_turbine_dataset1.csv") # Make sure this CSV is in the same folder

# Features & Target
X = df[["Air_Density", "Temperature", "Humidity", "Blade_Length"]]
y = df["Power_Output"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "wind_model.pkl")
print("✅ Model trained and saved as wind_model.pkl")
