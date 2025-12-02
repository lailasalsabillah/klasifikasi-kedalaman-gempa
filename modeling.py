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


def load_dataset():
    """
    Load dataset dari file dataset_gempa.csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset-gempa.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset dimuat dengan shape: {df.shape}")

    return df


def prepare_labels(df):
    """
    Siapkan label kelas kedalaman.

    - Jika sudah ada kolom 'depth_class' → langsung dipakai
    - Jika hanya ada kolom 'depth' → dibuat 3 kelas:
        0: shallow   (< 70 km)
        1: intermediate (70–300 km)
        2: deep     (> 300 km)
    """
    if "depth_class" in df.columns:
        df["depth_class"] = df["depth_class"].astype(int)
        print("Menggunakan kolom 'depth_class' sebagai label.")
    elif "depth" in df.columns:
        print("Membuat label 'depth_class' berdasarkan kolom 'depth'.")
        bins = [0, 70, 300, 1e9]
        labels = [0, 1, 2]
        df["depth_class"] = pd.cut(
            df["depth"], bins=bins, labels=labels, right=False
        ).astype(int)
    else:
        raise ValueError(
            "Dataset harus memiliki kolom 'depth_class' atau 'depth'."
        )

    return df


def select_features(df):
    """
    Memilih kolom fitur utama yang akan digunakan model.

    Daftar kandidat fitur:
    - year, latitude, longitude, mag
    - gap, dmin, rms
    - horizontalError, depthError, magError
    """
    candidate_cols = [
        "year",
        "latitude",
        "longitude",
        "mag",
        "gap",
        "dmin",
        "rms",
        "horizontalError",
        "depthError",
        "magError",
    ]

    feature_cols = [c for c in candidate_cols if c in df.columns]

    if not feature_cols:
        raise ValueError(
            "Tidak ada kolom fitur yang ditemukan. "
            "Pastikan nama kolom di dataset sesuai dengan: "
            f"{candidate_cols}"
        )

    print("Fitur yang digunakan:", feature_cols)
    return df[feature_cols].values, feature_cols


def ensure_models_dir():
    """
    Memastikan folder 'models/' ada.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    return models_dir


def train_xgboost(X_train, y_train, X_test, y_test, models_dir, num_classes):
    """
    Training model XGBoost untuk klasifikasi multi-kelas.
    """
    print("\n=== Training XGBoost Classifier ===")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    xgb_path = os.path.join(models_dir, "xgb_depth_class.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"Model XGBoost disimpan ke: {xgb_path}")

    return xgb_model


def train_lstm(X_train, y_train, X_test, y_test, models_dir, num_classes):
    """
    Training model LSTM sederhana untuk klasifikasi.

    Di sini fitur dianggap sebagai sequence:
    - timesteps = jumlah fitur
    - features per timestep = 1
    """
    print("\n=== Training LSTM Model ===")

    # reshape ke bentuk (samples, timesteps, features)
    n_features = X_train.shape[1]
    X_train_seq = X_train.reshape((X_train.shape[0], n_features, 1))
    X_test_seq = X_test.reshape((X_test.shape[0], n_features, 1))

    # one-hot encoding label
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    model = Sequential(
        [
            Input(shape=(n_features, 1)),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train_seq,
        y_train_cat,
        epochs=30,
        batch_size=32,
        validation_data=(X_test_seq, y_test_cat),
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"\nLSTM Test Accuracy: {test_acc:.4f}")

    lstm_path = os.path.join(models_dir, "lstm_depth_class.keras")
    model.save(lstm_path)
    print(f"Model LSTM disimpan ke: {lstm_path}")

    return model, history


def main():
    # 1. Load dataset
    df = load_dataset()

    # 2. Siapkan label
    df = prepare_labels(df)

    # 3. Pilih fitur
    X, feature_cols = select_features(df)

    # 4. Ambil label y
    y = df["depth_class"].astype(int).values

    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir = ensure_models_dir()
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler disimpan ke: {scaler_path}")

    # 6. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    num_classes = len(np.unique(y))

    # 7. Train XGBoost
    train_xgboost(X_train, y_train, X_test, y_test, models_dir, num_classes)

    # 8. Train LSTM (opsional, tetapi tetap dibuat di sini)
    train_lstm(X_train, y_train, X_test, y_test, models_dir, num_classes)

    print("\n=== Selesai Training Semua Model ===")
    print("Fitur yang digunakan:", feature_cols)


if __name__ == "__main__":
    main()
