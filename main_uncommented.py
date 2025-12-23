import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("loan_data.csv")

df = df.drop(columns=["LoanID"])

X = df.drop(columns=["Default"])
y = df["Default"]

numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore",
                "MonthsEmployed", "NumCreditLines", "InterestRate",
                "LoanTerm", "DTIRatio"]

categorical_cols = ["Education", "EmploymentType", "MaritalStatus",
                    "HasMortgage", "HasDependents", "LoanPurpose",
                    "HasCoSigner"]

encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

X_encoded = encoder.fit_transform(X)

sm = SMOTE()
X_balanced, y_balanced = sm.fit_resample(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced
)

print("Fitting data...")
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("Predicting...")
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("ROC AUC:", roc_auc_score(y_test, proba))

def prompt_int(name):
    while True:
        try:
            return int(input(f"{name}: "))
        except ValueError:
            print("Enter an integer.")

def prompt_float(name):
    while True:
        try:
            return float(input(f"{name}: "))
        except ValueError:
            print("Enter a number.")

def prompt_yes_no(name):
    while True:
        s = input(f"{name} (Yes/No): ").strip()
        if s in ("Yes", "No"):
            return s
        print("Type Yes or No.")

def prompt_category(name):
    return input(f"{name}: ").strip()

def get_user_row():
    row = {}

    row["Age"] = prompt_int("Age")
    row["Income"] = prompt_float("Income (eg 55000)")
    row["LoanAmount"] = prompt_float("LoanAmount (eg 120000)")
    row["CreditScore"] = prompt_int("CreditScore (300-850)")
    row["MonthsEmployed"] = prompt_int("MonthsEmployed")
    row["NumCreditLines"] = prompt_int("NumCreditLines")
    row["InterestRate"] = prompt_float("InterestRate (percent, eg 7.5)")
    row["LoanTerm"] = prompt_int("LoanTerm (months, eg 36)")
    row["DTIRatio"] = prompt_float("DebtToIncomeRatio (eg 0.35)")

    row["Education"] = prompt_category("Education (High School, Bachelor's, Master's, PhD)")
    row["EmploymentType"] = prompt_category("EmploymentType (Full-time, Part-time, Self-employed, Unemployed)")
    row["MaritalStatus"] = prompt_category("MaritalStatus (Single, Married, Divorced)")
    row["LoanPurpose"] = prompt_category("LoanPurpose (Auto, Home, Business, Education, Other)")

    row["HasMortgage"] = prompt_yes_no("HasMortgage")
    row["HasDependents"] = prompt_yes_no("HasDependents")
    row["HasCoSigner"] = prompt_yes_no("HasCoSigner")

    return pd.DataFrame([row])


while True:
    print("\nEnter a loan to predict")
    user_df = get_user_row()
    user_encoded = encoder.transform(user_df)

    p_default = model.predict_proba(user_encoded)[:, 1][0]
    yhat = int(p_default >= 0.5)

    print("\nResult:")
    print("Probability (Default > 0.5):", round(p_default, 4))
    print("Predicted class:", yhat, "(1=default, 0=no default)")

    again = input("\nAnother customer? (y/n): ").strip().lower()
    if again != "y":
        break