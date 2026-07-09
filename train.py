import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("cleaned_dataset.csv")
df = df.dropna()
df = df.drop_duplicates()

# Convert Debt and Income to rupees only if still small scale
if df["Debt"].max() < 1000:
    df["Debt"] = df["Debt"] * 100000

if df["Income"].max() < 100000:
    df["Income"] = df["Income"] * 1000

# Convert CreditScore from old 0-67 scale to 300-800 scale
if df["CreditScore"].max() <= 100:
    df["CreditScore"] = 300 + (df["CreditScore"] / 67) * 500

# Add extra features if missing
if "EmploymentStatus" not in df.columns:
    df["EmploymentStatus"] = df["YearsEmployed"].apply(
        lambda x: 1 if float(x) > 0 else 0
    )
else:
    df["EmploymentStatus"] = df["EmploymentStatus"].astype(str).str.lower().map({
        "employed": 1,
        "self-employed": 1,
        "student": 0,
        "unemployed": 0,
        "retired": 0
    }).fillna(0)

if "Citizenship" not in df.columns:
    if "Citizen" in df.columns:
        df["Citizenship"] = df["Citizen"]
    else:
        df["Citizenship"] = 1
else:
    df["Citizenship"] = df["Citizenship"].astype(str).str.lower().map({
        "indian": 1,
        "nri": 1,
        "other": 0
    }).fillna(1)

# Industry safe conversion
df["Industry"] = pd.to_numeric(df["Industry"], errors="coerce").fillna(0)

features = [
    "Gender",
    "Age",
    "Debt",
    "Married",
    "BankCustomer",
    "Industry",
    "YearsEmployed",
    "EmploymentStatus",
    "Citizenship",
    "PriorDefault",
    "CreditScore",
    "Income"
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Approved"] = pd.to_numeric(df["Approved"], errors="coerce")

df = df.dropna()

X = df[features]
y = df["Approved"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

model_data = {
    "model": model,
    "columns": features
}

with open("model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Model saved successfully")