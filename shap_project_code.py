
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import shap

# Load your dataset (replace with actual CSV)
# df = pd.read_csv("credit_risk.csv")

# Dummy example dataset
np.random.seed(0)
df = pd.DataFrame({
    "age": np.random.randint(20,60,200),
    "income": np.random.randint(20000,100000,200),
    "loan_amount": np.random.randint(1000,20000,200),
    "default": np.random.randint(0,2,200)
})

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, pred)
f1 = f1_score(y_test, (pred>0.5).astype(int))

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print("AUC:", auc)
print("F1 Score:", f1)

# Global Importance
importance = np.abs(shap_values.values).mean(axis=0)
print("Global SHAP Importance:", dict(zip(X.columns, importance)))
