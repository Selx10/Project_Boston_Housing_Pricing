import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#Load the model 
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print("Input JSON :",data)
    input_data = np.array(list(data.values())).reshape(1,-1)
    print("Input Data Array :", input_data)

    new_data = scaler.transform(input_data)
    print("Input Data :", new_data)
    output = regmodel.predict(new_data)
    print("Prediction :",output[0])
    return jsonify({'prediction':float(output[0])})

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)
    return render_template("home.html",prediction_text = "The Predicted value is{}".format(output))


if __name__ == "__main__":
    app.run(debug=True)