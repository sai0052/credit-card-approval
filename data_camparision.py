import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("cleaned_dataset.csv")

print("\nDataset Loaded Successfully")
print("--------------------------------")


# ==========================
# REMOVE MISSING VALUES
# ==========================

df = df.dropna()

print("Missing values removed")


# ==========================
# ENCODE CATEGORICAL DATA
# ==========================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns


encoder = LabelEncoder()


for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])


print("Categorical encoding completed")


# ==========================
# FEATURES AND TARGET
# ==========================

print("\nAvailable Columns:")
print(df.columns)


# CHANGE ONLY IF YOUR COLUMN NAME IS DIFFERENT

target = "Approved"


X = df.drop(target, axis=1)

y = df[target]


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# ==========================
# FEATURE SCALING
# ==========================

scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


print("Feature scaling completed")


# ==========================
# MODELS
# ==========================

models = {


"Logistic Regression":

LogisticRegression(
    max_iter=5000,
    solver="lbfgs"
),


"Decision Tree":

DecisionTreeClassifier(
    random_state=42
),



"Random Forest":

RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

}



results = {}



# ==========================
# TRAINING AND TESTING
# ==========================


for name, model in models.items():


    print("\n")
    print("="*60)
    print(name.upper())
    print("="*60)


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    results[name] = accuracy



    print("\nAccuracy:")

    print(
        round(
            accuracy*100,
            2
        ),
        "%"
    )



    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )



    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            prediction
        )
    )



# ==========================
# FINAL COMPARISON
# ==========================


print("\n")

print("="*60)

print("FINAL MODEL ACCURACY COMPARISON")

print("="*60)



for model, score in results.items():

    print(
        model,
        " : ",
        round(score*100,2),
        "%"
    )



# ==========================
# BEST MODEL
# ==========================


best_model = max(
    results,
    key=results.get
)


print("\n")

print("BEST MODEL:")

print(best_model)


print(
    "BEST ACCURACY:",
    round(
        results[best_model]*100,
        2
    ),
    "%"
)