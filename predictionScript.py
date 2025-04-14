import joblib
import numpy as np

# Load the model
model = joblib.load("beegfs_chunk_model.pkl")

# Define test input (use same order as training: File Size, DD Write, DD Read, FIO Write, FIO Read)
test_metrics = {
    "File Size": 500,
    "DD Write": 650,
    "DD Read": 620,
    "FIO Write": 600,
    "FIO Read": 590
}

# Convert to 2D array for prediction
features = np.array([[test_metrics[col] for col in ['File Size', 'DD Write', 'DD Read', 'FIO Write', 'FIO Read']]])

# Make prediction
predicted_chunk_size = model.predict(features)
print(f"Predicted Chunk Size: {predicted_chunk_size[0]:.2f} MB")