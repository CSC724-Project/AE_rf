import subprocess
import sys

# 🔍 Ensure required packages are installed
required_packages = ["pandas", "numpy", "scikit-learn", "joblib"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"📦 '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ✅ After ensuring packages, import them
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class BeeGFSAnalysisEngine:
    def __init__(self, data_csv, model_path="beegfs_chunk_model.pkl"):
        self.data_csv = data_csv
        self.model_path = model_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_columns = ['File Size', 'DD Write', 'DD Read', 'FIO Write', 'FIO Read']

    def load_and_train_model(self):
        print("🔄 Loading training data...")
        try:
            df = pd.read_csv(self.data_csv)
            print(f"✅ Data loaded: {len(df)} rows.")
        except Exception as e:
            logging.error("❌ Failed to load data from CSV: %s", e)
            print("❌ Error loading data.")
            return

        print("🧹 Cleaning data...")
        df = df.dropna()
        for col in self.feature_columns + ['Chunk Size']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        if df.empty:
            logging.error("❌ No valid data after cleaning.")
            print("❌ No valid data available for training.")
            return

        print("🔧 Preparing features and labels...")
        X = df[self.feature_columns]
        y = df['Chunk Size']

        print("✂️ Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print("🌲 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        logging.info("✅ Model trained. R2 score: %.3f", score)
        print(f"✅ Model training complete. R2 score: {score:.3f}")

        print("💾 Saving trained model...")
        joblib.dump(self.model, self.model_path)
        print(f"✅ Model saved to: {self.model_path}")

    def predict_chunk_size(self, metrics):
        print("🔍 Predicting chunk size for input metrics...")
        try:
            X_new = np.array([[metrics[feature] for feature in self.feature_columns]])
        except KeyError as e:
            logging.error("❌ Missing metric key: %s", e)
            print(f"❌ Error: Missing metric key: {e}")
            return None

        predicted_chunk_size = self.model.predict(X_new)
        print(f"🎯 Predicted optimal chunk size: {predicted_chunk_size[0]:.2f}")
        return predicted_chunk_size[0]

    def update_beegfs_configuration(self, chunk_size):
        cmd = f'beegfs-tool --set-chunk-size {chunk_size:.2f}'
        print(f"🛠️ Running system command: {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
            print("✅ BeeGFS configuration updated successfully.")
            logging.info("BeeGFS configuration updated: %s", result.stdout)
        except subprocess.CalledProcessError as error:
            logging.error("❌ BeeGFS configuration failed: %s", error.stderr)
            print("❌ Error updating BeeGFS configuration.")

    def run(self, metrics):
        print("🚀 Running BeeGFS Analysis Engine...")
        optimal_chunk_size = self.predict_chunk_size(metrics)
        if optimal_chunk_size is not None:
            self.update_beegfs_configuration(optimal_chunk_size)
        else:
            print("⚠️ Prediction failed. Configuration update skipped.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="BeeGFS Analysis Engine CLI")
    parser.add_argument("--data", required=True, help="Path to training CSV file")
    parser.add_argument("--predict", action="store_true", help="Run prediction and update")
    parser.add_argument("--file_size", type=float, help="File size in MB")
    parser.add_argument("--dd_write", type=float, help="DD Write throughput")
    parser.add_argument("--dd_read", type=float, help="DD Read throughput")
    parser.add_argument("--fio_write", type=float, help="FIO Write bandwidth")
    parser.add_argument("--fio_read", type=float, help="FIO Read bandwidth")
    args = parser.parse_args()

    engine = BeeGFSAnalysisEngine(data_csv=args.data)

    print("📚 Training model...")
    engine.load_and_train_model()

    if args.predict:
        print("🧠 Running prediction...")
        required_metrics = {
            "File Size": args.file_size,
            "DD Write": args.dd_write,
            "DD Read": args.dd_read,
            "FIO Write": args.fio_write,
            "FIO Read": args.fio_read
        }

        missing = [k for k, v in required_metrics.items() if v is None]
        if missing:
            print(f"⚠️ Missing required inputs for prediction: {missing}")
        else:
            engine.run(required_metrics)