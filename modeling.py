# ============================================================
# modeling.py
# Training model klasifikasi kedalaman gempa
# Menggunakan LSTM & XGBoost
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier
import joblib
import tensorflow as tf

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("gempabumi-fiks.csv")

print("Dataset loaded:", df.shape)

# ============================================================
# 2. FEATURE SELECTION
# ============================================================

feature_cols = [
    "year", "latitude", "longitude", "mag",
    "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError"
]

X = df[feature_cols]
y = df["depth_class"]

# ============================================================
# 3. TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ============================================================
# 4. CLASS WEIGHTS (IMBALANCE HANDLING)
# ============================================================

classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, cw))

print("Class weights:", class_weights)

# ============================================================
# 5. SCALING DATA UNTUK LSTM
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# reshape untuk LSTM (samples, time_steps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat  = to_categorical(y_test, num_classes=num_classes)

# ============================================================
# 6. TRAIN MODEL LSTM
# ============================================================

tf.random.set_seed(42)

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train_lstm.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model_lstm.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model_lstm.fit(
    X_train_lstm, y_train_cat,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 7. EVALUASI LSTM
# ============================================================

y_pred_lstm = np.argmax(model_lstm.predict(X_test_lstm), axis=1)

print("\n=== LSTM EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm, digits=4))

# ============================================================
# 8. TRAIN MODEL XGBOOST
# ============================================================

sample_weight = y_train.map(class_weights)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train, sample_weight=sample_weight)

y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBOOST EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, digits=4))

# ============================================================
# 9. SAVE MODELS & SCALER
# ============================================================

os.makedirs("models", exist_ok=True)

model_lstm.save("models/lstm_depth_class.keras")
joblib.dump(xgb, "models/xgb_depth_class.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels saved successfully in /models/")
