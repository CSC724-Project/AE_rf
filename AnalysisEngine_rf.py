import pandas as pd
import numpy as np
import subprocess
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class BeeGFSAnalysisEngine:
    """
    A BeeGFS Analysis Engine that:
    1. Loads historical access metrics.
    2. Trains a Random Forest model to predict the optimal chunk size.
    3. Predicts an optimal chunk size for newly collected metrics.
    4. Updates the BeeGFS configuration using the predicted chunk size.
    """

    def __init__(self, data_csv):
        """
        Initializes the engine with the path to the training CSV data.
        """
        self.data_csv = data_csv
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Define the feature columns that match your CSV structure.
        self.feature_columns = ['file_size', 'io_rate', 'access_latency']

    def load_and_train_model(self):
        """
        Loads data from CSV and trains the Random Forest model.
        Splits the data into training and test sets and logs the R2 score.
        """
        try:
            df = pd.read_csv(self.data_csv)
        except Exception as e:
            logging.error("Failed to load data from CSV: %s", e)
            return

        # Ensure the CSV has the necessary columns.
        for col in self.feature_columns + ['optimal_chunk_size']:
            if col not in df.columns:
                logging.error("Missing required column '%s' in the CSV data.", col)
                return

        X = df[self.feature_columns]
        y = df['optimal_chunk_size']

        # Split data for training and validation.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train the model.
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        logging.info("Random Forest Model Trained with R2 score: %.3f", score)

    def predict_chunk_size(self, metrics):
        """
        Predicts the optimal chunk size given a dictionary of metrics.
        
        Parameters:
            metrics (dict): Dictionary with keys matching self.feature_columns.
        
        Returns:
            float: Predicted optimal chunk size.
        """
        try:
            # Create a 2D array for a single sample.
            X_new = np.array([[metrics[feature] for feature in self.feature_columns]])
        except KeyError as e:
            logging.error("Missing metric key: %s", e)
            return None

        predicted_chunk_size = self.model.predict(X_new)
        logging.info("Predicted optimal chunk size: %.2f", predicted_chunk_size[0])
        return predicted_chunk_size[0]

    def update_beegfs_configuration(self, chunk_size):
        """
        Applies the new chunk size configuration.
        This method simulates using a system command to apply the change.
        
        Parameters:
            chunk_size (float): The predicted chunk size.
        """
        # Construct the command to update BeeGFS configuration.
        # Replace this with your actual command or API call.
        cmd = f'beegfs-tool --set-chunk-size {chunk_size:.2f}'

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
            logging.info("BeeGFS configuration updated successfully: %s", result.stdout)
        except subprocess.CalledProcessError as error:
            logging.error("Error updating BeeGFS configuration: %s", error.stderr)

    def run(self, metrics):
        """
        Executes the full pipeline:
          1. Predicts the optimal chunk size.
          2. Updates the BeeGFS configuration.
        
        Parameters:
            metrics (dict): New file access metrics.
        """
        optimal_chunk_size = self.predict_chunk_size(metrics)
        if optimal_chunk_size is not None:
            self.update_beegfs_configuration(optimal_chunk_size)


if __name__ == "__main__":
    # Set up logging configuration.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Path to the CSV file containing historical metrics.
    training_csv = "access_logs.csv"
    engine = BeeGFSAnalysisEngine(training_csv)

    # Step 1: Load the historical data and train the model.
    engine.load_and_train_model()

    # Step 2: Collect new metrics.
    # In a real scenario, this dictionary should be built from
    # current file system and performance measurements.
    new_metrics = {
        'file_size': 500,        # e.g., in MB
        'io_rate': 50,           # e.g., in MB/s
        'access_latency': 0.1    # e.g., in seconds
    }

    # Step 3: Run the analysis engine to predict and apply the new chunk size.
    engine.run(new_metrics)