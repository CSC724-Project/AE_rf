import pandas as pd
import numpy as np
import subprocess
import logging
import joblib

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
        try:
            df = pd.read_csv(self.data_csv)
        except Exception as e:
            logging.error("Failed to load data from CSV: %s", e)
            return

        df = df.dropna()
        for col in self.feature_columns + ['Chunk Size']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        X = df[self.feature_columns]
        y = df['Chunk Size']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        logging.info("Random Forest Model Trained with R2 score: %.3f", score)

        joblib.dump(self.model, self.model_path)
        logging.info("Model saved to: %s", self.model_path)

    def predict_chunk_size(self, metrics):
        try:
            X_new = np.array([[metrics[feature] for feature in self.feature_columns]])
        except KeyError as e:
            logging.error("Missing metric key: %s", e)
            return None

        predicted_chunk_size = self.model.predict(X_new)
        logging.info("Predicted optimal chunk size: %.2f", predicted_chunk_size[0])
        return predicted_chunk_size[0]

    def update_beegfs_configuration(self, chunk_size):
        cmd = f'beegfs-tool --set-chunk-size {chunk_size:.2f}'
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
            logging.info("BeeGFS configuration updated: %s", result.stdout)
        except subprocess.CalledProcessError as error:
            logging.error("BeeGFS configuration failed: %s", error.stderr)

    def run(self, metrics):
        optimal_chunk_size = self.predict_chunk_size(metrics)
        if optimal_chunk_size is not None:
            self.update_beegfs_configuration(optimal_chunk_size)