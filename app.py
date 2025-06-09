from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('crop_price_prediction_model.pkl', 'rb'))

# Load the actual feature columns used during training
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Optional: List of states and crops to show on the form
states = ['Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka', 'Madhya Pradesh',
          'Maharashtra', 'Orissa', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Uttar Pradesh', 'West Bengal']

crops = ['ARHAR', 'COTTON', 'GRAM', 'GROUNDNUT', 'MAIZE', 'MOONG', 'PADDY', 'MUSTARD', 'SUGARCANE', 'WHEAT']

@app.route('/')
def home():
    return render_template('index.html', states=states, crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    state = request.form['state']
    crop = request.form['crop']
    cost = float(request.form['cost'])
    production = float(request.form['production'])
    yield_amt = float(request.form['yield'])
    temp = float(request.form['temperature'])
    rainfall = float(request.form['rainfall'])

    # Step 1: Start input dictionary with numerical values
    input_data = {
        'CostCultivation': cost,
        'Production': production,
        'Yield': yield_amt,
        'Temperature': temp,
        'RainFall Annual': rainfall
    }

    # Step 2: Initialize missing one-hot columns with 0
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0

    # Step 3: Set correct one-hot flags
    state_col = f'State_{state}'
    crop_col = f'Crop_{crop}'
    if state_col in model_columns:
        input_data[state_col] = 1
    if crop_col in model_columns:
        input_data[crop_col] = 1

    # Step 4: Prepare input DataFrame in the right column order
    input_df = pd.DataFrame([input_data], columns=model_columns)

    # Step 5: Predict
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
