# Import Libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, flash
import pickle

# Create Flask app
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

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
    try:
    	features = [float(i) for i in request.form.values()]
    except:
    	flash("Please enter numbers.")
    	return render_template('index.html')
    features = np.array([features])

    # prediction
    prediction = model.predict(features)
    gender = index2label[prediction[0]]

    prediction_text = "Predicted as {}.".format(gender)

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)