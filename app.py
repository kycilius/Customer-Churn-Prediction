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
    form_data = {}

    if request.method == "POST":

        if model is None:
            return "Model not loaded"

        form_data = request.form.to_dict()

        input_df = pd.DataFrame([{
            "CreditScore": int(form_data["CreditScore"]),
            "Geography": form_data["Geography"],
            "Gender": form_data["Gender"],
            "Age": int(form_data["Age"]),
            "Tenure": int(form_data["Tenure"]),
            "Balance": float(form_data["Balance"]),
            "NumOfProducts": int(form_data["NumOfProducts"]),
            "HasCrCard": int(form_data["HasCrCard"]),
            "IsActiveMember": int(form_data["IsActiveMember"]),
            "EstimatedSalary": float(form_data["EstimatedSalary"]),
            "Card Type": form_data["Card Type"]
        }])

        prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        form_data=form_data
    )
