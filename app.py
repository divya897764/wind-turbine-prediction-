from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "wind_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Page (GET → show form, POST → make prediction)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Read inputs explicitly
            air_density = float(request.form.get('air_density', 1.225))
            temperature = float(request.form.get('temperature', 20.0))
            humidity = float(request.form.get('humidity', 50.0))
            blade_length = float(request.form.get('blade_length', 45.0))

            # Prepare features for ML model
            features = np.array([[air_density, temperature, humidity, blade_length]])
            pred = model.predict(features)[0]

            return render_template(
                'prediction.html',
                prediction_text=f"Predicted Power Output: {pred:.2f} units"
            )
        except Exception as e:
            return render_template('prediction.html', prediction_text=f"Error: {str(e)}")
    else:
        # When user first opens the prediction page
        return render_template('prediction.html')


if __name__ == "__main__":
    print("🚀 Starting Flask server... Open http://127.0.0.1:5000/")
    app.run(debug=True)
