import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# prediction labels dict
index2label = {0: "Female", 1: "Male"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # preprocess given features
    features = [float(i) for i in request.form.values()]
    features = np.array([features])

    # prediction
    prediction = model.predict(features)
    gender = index2label[prediction[0]]

    prediction_text = "Predicted as {}.".format(gender)

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)