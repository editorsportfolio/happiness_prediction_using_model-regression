from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Load prepared model
model = pickle.load(open("happiness_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[data["gdp"], data["social"], data["health"], data["freedom"]]]
    prediction = model.predict(features)[0]
    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)