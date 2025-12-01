import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# === LOAD DATASET ===
df = pd.read_csv("dataset/Dataset_gempabumi.csv")
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["year"] = df["time"].dt.year

# filter tahun
df = df[(df["year"] >= 2020) & (df["year"] <= 2024)].reset_index(drop=True)

# buat kolom kedalaman kelas
def depth_to_class(depth):
    if depth < 70:
        return 0
    elif depth < 300:
        return 1
    else:
        return 2

df["depth_class"] = df["depth"].apply(depth_to_class)

# pilih fitur
feature_cols = [
    "latitude", "longitude", "mag",
    "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError",
    "year"
]

X = df[feature_cols]
y = df["depth_class"]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === TRAINING XGBOOST ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=5,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# === SAVE MODEL ===
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(model, "models/xgb_depth_class.pkl")

print("Model & scaler saved successfully!")
