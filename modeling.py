import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

from imblearn.over_sampling import SMOTE


# ================================
# 1. Load Dataset
# ================================
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset-gempa.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError("dataset-gempa.csv tidak ditemukan!")

    df = pd.read_csv(data_path)
    print("Dataset loaded:", df.shape)
    return df


# ================================
# 2. Siapkan Label
# ================================
def prepare_labels(df):
    if "depth_class" not in df.columns:
        raise ValueError("Kolom depth_class tidak ditemukan!")

    df["depth_class"] = df["depth_class"].astype(int)
    print("Label distribution:\n", df["depth_class"].value_counts())

    return df


# ================================
# 3. Pilih Fitur Sesuai Dataset Kamu
# ================================
def select_features(df):
    feature_cols = [
        "year", "latitude", "longitude", "mag",
        "gap", "dmin", "rms",
        "horizontalError", "depthError", "magError"
    ]

    X = df[feature_cols].values
    print("Features used:", feature_cols)
    return X, feature_cols


# ================================
# BUAT FOLDER MODEL
# ================================
def ensure_models_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


# ================================
# 4. Train XGBoost dengan SMOTE
# ================================
def train_xgb(X_train, y_train, X_test, y_test, models_dir, num_classes):

    print("\n=== Training XGBoost (with balanced data) ===")

    xgb = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))

    xgb_path = os.path.join(models_dir, "xgb_depth_class.pkl")
    joblib.dump(xgb, xgb_path)
    print("XGBoost saved to:", xgb_path)

    return xgb


# ================================
# 5. Train LSTM
# ================================
def train_lstm(X_train, y_train, X_test, y_test, models_dir, num_classes):

    print("\n=== Training LSTM ===")

    timesteps = X_train.shape[1]
    X_train_seq = X_train.reshape((X_train.shape[0], timesteps, 1))
    X_test_seq = X_test.reshape((X_test.shape[0], timesteps, 1))

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    model = Sequential([
        Input(shape=(timesteps, 1)),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train_seq, y_train_cat,
        epochs=25,
        batch_size=32,
        validation_data=(X_test_seq, y_test_cat),
        verbose=1
    )

    lstm_path = os.path.join(models_dir, "lstm_depth_class.keras")
    model.save(lstm_path)
    print("LSTM saved to:", lstm_path)

    return model


# ================================
# MAIN TRAINING PIPELINE
# ================================
def main():

    # 1. Load dataset
    df = load_dataset()

    # 2. Label handling
    df = prepare_labels(df)

    # 3. Select fitur
    X, feature_cols = select_features(df)
    y = df["depth_class"].values

    # 4. Scale data (SCALER FIT SEBELUM SMOTE → lebih stabil)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir = ensure_models_dir()
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print("Scaler saved to:", scaler_path)

    # 5. FIX IMBALANCE (SMOTE)
    print("\n=== Applying SMOTE Oversampling ===")
    sm = SMOTE(random_state=42)
    X_balanced, y_balanced = sm.fit_resample(X_scaled, y)
    print("Balanced shape:", X_balanced.shape)

    # 6. Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    num_classes = len(np.unique(y))

    # 7. Train XGBoost
    train_xgb(X_train, y_train, X_test, y_test, models_dir, num_classes)

    # 8. Train LSTM
    train_lstm(X_train, y_train, X_test, y_test, models_dir, num_classes)

    print("\n=== TRAINING COMPLETE! ===")


if __name__ == "__main__":
    main()
