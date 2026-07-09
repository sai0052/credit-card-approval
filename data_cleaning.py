import pandas as pd

df = pd.read_csv("dataset.csv")

df = df.dropna()
df = df.drop_duplicates()

# Convert original dataset scale to rupees
df["Income"] = df["Income"] * 1000
df["Debt"] = df["Debt"] * 100000

# Clean Industry column safely
industry_map = {
    "industrials": "Manufacturing",
    "information technology": "IT",
    "technology": "IT",
    "financials": "Banking",
    "finance": "Banking",
    "health care": "Healthcare",
    "healthcare": "Healthcare",
    "education": "Education",
    "government": "Government",
    "business": "Business",
    "other": "Other"
}

if "Industry" in df.columns:
    df["Industry"] = df["Industry"].astype(str).str.strip().str.lower()
    df["Industry"] = df["Industry"].map(industry_map).fillna("Other")
else:
    df["Industry"] = "Other"

df["EmploymentStatus"] = df["YearsEmployed"].apply(
    lambda x: "Employed" if float(x) > 0 else "Unemployed"
)

df["Citizenship"] = "Indian"

df.to_csv("cleaned_dataset.csv", index=False)

print("Data cleaning completed successfully")
print(df[["Debt", "Income", "Industry", "EmploymentStatus", "Citizenship"]].head())