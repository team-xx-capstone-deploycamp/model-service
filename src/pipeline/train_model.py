import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from minio import Minio
import io

# Konfigurasi MinIO
MINIO_ENDPOINT = "minio.capstone.pebrisulistiyo.com"  # Ganti dengan endpoint MinIO kamu
MINIO_ACCESS_KEY = "YOUR_ACCESS_KEY"  
MINIO_SECRET_KEY = "YOUR_SECRET_KEY"  
BUCKET_NAME = "bucket"

# Inisialisasi client MinIO
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True
)

# Ambil data dari MinIO
def load_data(filename="used_car_cleaned.csv"):
    print(f"📥 Mengunduh dataset '{filename}' dari bucket '{BUCKET_NAME}'...")
    response = client.get_object(BUCKET_NAME, filename)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    print("✅ Dataset berhasil diunduh.")
    return df

# Preprocessing data
def preprocess_data(df):
    print("⚙️ Memproses data...")
    # Pisahkan fitur dan target
    if "price" not in df.columns:
        raise ValueError("Kolom 'price' tidak ditemukan di dataset.")

    X = df.drop(columns=["price"])
    y = df["price"]

    # Label encoding untuk kolom kategorikal
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✅ Data siap. Total data latih: {len(X_train)}, data uji: {len(X_test)}")
    return X_train, X_test, y_train, y_test, label_encoders

# Training model
def train_model(X_train, y_train):
    print("🚀 Memulai training model XGBoost...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Model berhasil dilatih.")
    return model

# Simpan model ke MinIO
def save_model(model, label_encoders, filename="car_price_model.pkl"):
    print(f"💾 Menyimpan model ke MinIO sebagai '{filename}'...")
    buffer = io.BytesIO()
    joblib.dump({"model": model, "encoders": label_encoders}, buffer)
    buffer.seek(0)
    client.put_object(
        BUCKET_NAME, filename, buffer, length=buffer.getbuffer().nbytes
    )
    print("✅ Model berhasil disimpan ke MinIO.")

# Pipeline utama
def run_pipeline():
    df = load_data("used_car_cleaned.csv")  # Gunakan dataset yang sudah dibersihkan
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df)
    model = train_model(X_train, y_train)
    save_model(model, encoders, "car_price_model.pkl")
    print("🎯 Pipeline selesai.")

if __name__ == "__main__":
    run_pipeline()
