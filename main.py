# To store and manage our data.
import pandas as pd
# To split the data into a training and testing set
from sklearn.model_selection import train_test_split
# For splitting our categorical data into into seperate features. 
from sklearn.preprocessing import OneHotEncoder
# Column Transformer is a preprocessor pipeline, applying
# specific transformations to selected columns only. 
from sklearn.compose import ColumnTransformer
# For evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Our model, picked for it's efficacy on mixed numerical and categorical features
# along with it's ability to pick up nonlinear patterns.
from sklearn.ensemble import GradientBoostingClassifier
# For balancing our dataset, as there are a lot more non defaults than defaults.
from imblearn.over_sampling import SMOTE



# Loads the data from the csv into a Pandas dataframe.
df = pd.read_csv("loan_data.csv")

# Drop the ID as we don't need to identify which loan is which.
# This could also cause the model to 'memorize' our training data,
# picking up useless patterns from the IDs.
df = df.drop(columns=["LoanID"])

# Make the target, whether the individual defaulted or not, it's own list.
# This makes it so the model doesn't get to 'cheat' by knowing if the
# individual defaulted or not and using that info to train.
X = df.drop(columns=["Default"])
y = df["Default"]

# List which features are numerical
numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore",
                "MonthsEmployed", "NumCreditLines", "InterestRate",
                "LoanTerm", "DTIRatio"]

# and which are categorical.
categorical_cols = ["Education", "EmploymentType", "MaritalStatus",
                    "HasMortgage", "HasDependents", "LoanPurpose",
                    "HasCoSigner"]

# Use one hot encoder to seperate the categorical features.
# Create our preprocessing pipeline for the categorical columns only,
# as we don't want to split the numerical categories. 
encoder = ColumnTransformer(
    transformers=[
        # Unknown categories at prediction time are ignored
        # "cat" basically is the name of this transformer.
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    # The columns that aren't categorical remain the same, and just pass through.
    remainder="passthrough"
)

# This line does two processes, fit and transform. Fit scans the data and sees
# what the transformation needs. For a one hot encoder, this is discovering
# all the categoires that appear in each categorical column.
# Transform then applies this transformation to the data, replacing each
# categorical feature with the one hot encoded matrix. 
X_encoded = encoder.fit_transform(X)

# SMOTE is a method to fight imbalance, which is when the minority (defaulting)
# class is rarer than the majority. If we don't balance our data, then the model
# will just predict that the input person will not default everytime. This is
# dangerous as it will seem like that the model is very accurate, as the testing
# data will not have many default instances.
# 
# What SMOTE (Synthetic Minority Oversampling Technique) does is it creates fake
# samples of the minority by interpolating the real minority data points. This 
# forces the model to learn more about both, not just that it is more likely to 
# not default than default. 
sm = SMOTE()
# This line does two steps as well, fit and resample. Fit finds the minority class,
# and resample constructs the synthetic minority examples from SMOTE and returns a
# new dataset where the minority and majority classes are balanced.
# x_balanced is the feature matrix after SMOTE and y_balanced 
X_balanced, y_balanced = sm.fit_resample(X_encoded, y)

# Splits the data into 80% training data, 20% testing data.
# Stratify makes sure we have the same ratio of both classes
# in the training and testing data. 
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced
)

print("Fitting data...")
# Create our model (gradient boosted decision tree)
model = GradientBoostingClassifier()
# Train the model with the training splits. 
model.fit(X_train, y_train)

print("Predicting...")
# Runs the model to predict based on our features in the X_test matrix.
pred = model.predict(X_test)
# Returns the probabilities of defaulting or not defaulting.
# [:, 1] is because it returns a matrix with 2 columns for either, and default
# is the second one.  
proba = model.predict_proba(X_test)[:, 1]

# Calculates precision of the model with the actual default / no default results.
print("Classification Report:")
print(classification_report(y_test, pred))
# A confusion matrix compares the actual result versus predicted result. This shows
# us false positives / negatives clearly.
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
# Area Under the Receiver Operating Characteristic Curve, for evaluating binary
# classifiers (default / no default). 1 represnts perfect, 0.5 represents random.
print("ROC AUC:", roc_auc_score(y_test, proba))

# Input new customers
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

    # Numerical
    row["Age"] = prompt_int("Age")
    row["Income"] = prompt_float("Income (eg 55000)")
    row["LoanAmount"] = prompt_float("LoanAmount (eg 120000)")
    row["CreditScore"] = prompt_int("CreditScore (300-850)")
    row["MonthsEmployed"] = prompt_int("MonthsEmployed")
    row["NumCreditLines"] = prompt_int("NumCreditLines")
    row["InterestRate"] = prompt_float("InterestRate (percent, eg 7.5)")
    row["LoanTerm"] = prompt_int("LoanTerm (months, eg 36)")
    row["DTIRatio"] = prompt_float("DebtToIncomeRatio (eg 0.35)")

    # Categorical
    row["Education"] = prompt_category("Education (High School, Bachelor's, Master's, PhD)")
    row["EmploymentType"] = prompt_category("EmploymentType (Full-time, Part-time, Self-employed, Unemployed)")
    row["MaritalStatus"] = prompt_category("MaritalStatus (Single, Married, Divorced)")
    row["LoanPurpose"] = prompt_category("LoanPurpose (Auto, Home, Business, Education, Other)")

    # Yes / No
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