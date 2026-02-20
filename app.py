from flask import Flask, render_template, request
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load model safely
try:
    model = joblib.load("best_churn_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":

        if model is None:
            return "Model not loaded"

        input_df = pd.DataFrame([{
            "CreditScore": int(request.form["CreditScore"]),
            "Geography": request.form["Geography"],
            "Gender": request.form["Gender"],
            "Age": int(request.form["Age"]),
            "Tenure": int(request.form["Tenure"]),
            "Balance": float(request.form["Balance"]),
            "NumOfProducts": int(request.form["NumOfProducts"]),
            "HasCrCard": int(request.form["HasCrCard"]),
            "IsActiveMember": int(request.form["IsActiveMember"]),
            "EstimatedSalary": float(request.form["EstimatedSalary"]),
            "Card Type": request.form["Card Type"]
        }])

        prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction)


# Do NOT run app here for Render
# Gunicorn will handle it
