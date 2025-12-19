import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ✅ Step 1: Load Dataset
print(" Step 1: Loading dataset...")
data = pd.read_csv("dataset.csv")   # your 500-row dataset
print(" Dataset loaded successfully")
print(data.head())
print(data.columns)


# ✅ Step 2: Features & Target
print(" Step 2: Splitting features and target...")
X = data.drop("fraud_risk", axis=1)
y = data["fraud_risk"]
print("Features and target created")


# ✅ Step 3: Train-Test Split
print("Step 3: Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(" Train-test split done")


#  Step 4: Train Model (FAST for low-end PC)
print(" Step 4: Training model...")
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train, y_train)
print(" Model trained successfully")


# Step 5: Test Accuracy
print(" Step 5: Testing model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(" Accuracy:", acc)
print("Saving model...")

import pickle

model_path = "fraud_model.pkl"   # same folder as train.py

file_obj = open(model_path, "wb")
pickle.dump(model, file_obj)
file_obj.close()

print("MODEL SAVED SUCCESSFULLY AT:")
print(model_path)