import os
import pickle
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
columns = model_data["columns"]


def encode_value(col, value):
    value = str(value).strip().lower()

    mappings = {
        "Gender": {"male": 1, "female": 0, "1": 1, "0": 0},
        "Married": {"yes": 1, "no": 0, "1": 1, "0": 0},
        "BankCustomer": {"yes": 1, "no": 0, "1": 1, "0": 0},

        # Important for your dataset
        "PriorDefault": {"no": 1, "yes": 0, "1": 1, "0": 0},

        "EmploymentStatus": {
            "unemployed": 0,
            "student": 0,
            "retired": 0,
            "employed": 1,
            "self-employed": 1
        },

        "Industry": {
            "other": 0,
            "it": 1,
            "banking": 2,
            "healthcare": 3,
            "education": 4,
            "government": 5,
            "business": 6,
            "manufacturing": 7
        },

        "Citizenship": {
            "indian": 1,
            "nri": 1,
            "other": 0
        }
    }

    if col in mappings:
        return mappings[col].get(value, 0)

    return float(value)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {}

        for required_field in ["Income", "Debt", "CreditScore"]:
            if not request.form.get(required_field):
                return f"Missing input field: {required_field}"

        raw_income = float(request.form.get("Income"))
        raw_debt = float(request.form.get("Debt"))
        raw_credit = float(request.form.get("CreditScore"))

        for col in columns:
            value = request.form.get(col)

            if value is None or value == "":
                return f"Missing input field: {col}"

            if col == "CreditScore":
                value = float(value)
                if value < 300 or value > 800:
                    return "Credit Score should be between 300 and 800"

            elif col in ["Debt", "Income", "Age", "YearsEmployed"]:
                value = float(value)

            else:
                value = encode_value(col, value)

            input_data[col] = value

        input_df = pd.DataFrame([input_data])
        input_df = input_df[columns]

        probability = model.predict_proba(input_df)[0]
        approved_probability = probability[1]
        confidence = round(max(probability) * 100, 2)

        prediction = 1 if approved_probability >= 0.60 else 0

        debt_income_ratio = raw_debt / raw_income if raw_income > 0 else 999

        # Strong business rules
        if raw_income <= 0:
            prediction = 0
            risk = "VERY HIGH RISK"
            message = "Income must be greater than zero."

        elif (
            raw_credit >= 700
            and input_data["PriorDefault"] == 1
            and raw_income >= 500000
            and debt_income_ratio <= 0.40
            and input_data["EmploymentStatus"] == 1
        ):
            prediction = 1
            risk = "LOW RISK"
            message = "Strong income, good credit score, stable employment, and no previous default."

        elif (
            raw_credit < 550
            or debt_income_ratio > 2
            or input_data["PriorDefault"] == 0
            or input_data["EmploymentStatus"] == 0
        ):
            prediction = 0
            risk = "HIGH RISK"
            message = "Low credit score, high debt, previous default, or unstable employment indicates risk."

        elif prediction == 0:
            risk = "MEDIUM RISK"
            message = "Model rejected this profile. Additional verification is recommended."

        else:
            risk = "LOW RISK"
            message = "Financial profile looks stable."

        if prediction == 1:
            result = "✅ CREDIT CARD APPROVED"
            status = "approved"
        else:
            result = "❌ CREDIT CARD REJECTED"
            status = "rejected"

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            risk=risk,
            message=message,
            status=status
        )

    except Exception as e:
        return f"Prediction Error: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)