import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset (replace with your dataset path)
df = pd.read_csv("loan_data.csv")

df.fillna(method='ffill', inplace=True)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
