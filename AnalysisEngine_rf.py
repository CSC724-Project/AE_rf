import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import KBinsDiscretizer

print("🚀 Starting model fine-tuning for BeeGFS chunk size prediction...")

# Load data
df = pd.read_csv("train.csv")  # Update path if needed
features = ['file_size_KB', 'access_count', 'avg_read_KB', 'avg_write_KB',
            'max_read_KB', 'max_write_KB', 'read_ops', 'write_ops', 'throughput_KBps']
target = 'chunk_size_KB'
df_clean = df[features + [target]].dropna()
print(f"✅ Loaded and cleaned data: {len(df_clean)} rows")

# Prepare data
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# For classification-style metrics
bin_enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_test_bins = bin_enc.fit_transform(y_test.values.reshape(-1, 1)).ravel()

# Parameter grid
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# RandomizedSearchCV
print("🔍 Starting hyperparameter search...")
base_model = RandomForestRegressor(random_state=42)
search = RandomizedSearchCV(estimator=base_model,
                            param_distributions=param_distributions,
                            n_iter=25,
                            cv=5,
                            verbose=1,
                            n_jobs=-1,
                            scoring='r2',
                            random_state=42)
search.fit(X_train, y_train)

# Evaluation
print("📈 Evaluating best model...")
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Classification-style metrics
y_pred_bins = bin_enc.transform(y_pred.reshape(-1, 1)).ravel()
accuracy = accuracy_score(y_test_bins, y_pred_bins)
precision = precision_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
recall = recall_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
f1 = f1_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)



# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
#     accuracy_score, precision_score, recall_score, f1_score
# )
# from sklearn.preprocessing import KBinsDiscretizer

# print("🚀 Starting BeeGFS Chunk Size Predictor...")

# # Load dataset
# csv_path = "train.csv"  # Change path if needed
# print(f"📂 Loading dataset from '{csv_path}'...")
# df = pd.read_csv(csv_path)
# print(f"✅ Loaded {len(df)} rows.")

# # Define columns
# features = ['file_size_KB', 'access_count', 'avg_read_KB', 'avg_write_KB',
#             'max_read_KB', 'max_write_KB', 'read_ops', 'write_ops', 'throughput_KBps']
# target = 'chunk_size_KB'

# print("🧹 Cleaning data...")
# df_clean = df[features + [target]].dropna()
# print(f"✅ Remaining rows after cleaning: {len(df_clean)}")

# # Prepare features and labels
# print("🧠 Preparing features and target variable...")
# X = df_clean[features]
# y = df_clean[target]

# # Split the data
# print("✂️ Splitting into train and test sets (70/30)...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}")

# # Train the model
# print("🌲 Training Random Forest Regressor...")
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# print("✅ Model training complete.")

# # Make predictions
# print("🔍 Making predictions on test data...")
# y_pred = model.predict(X_test)

# # Compute regression metrics
# print("📈 Calculating regression metrics...")
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)

# # Binned classification-style metrics
# print("🧪 Calculating classification-style metrics using binned chunk sizes...")
# bin_enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
# y_test_bins = bin_enc.fit_transform(y_test.values.reshape(-1, 1)).ravel()
# y_pred_bins = bin_enc.transform(y_pred.reshape(-1, 1)).ravel()

# accuracy = accuracy_score(y_test_bins, y_pred_bins)
# precision = precision_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
# recall = recall_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)
# f1 = f1_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0)

# Print all results
print("\nAll metrics computed successfully.")
print("\nPerformance Metrics:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.3f}")
print(f"MAPE: {mape:.3%}")

print("\nClassification Metrics (chunk size):")
print(f"Accuracy: {accuracy:.3%}")
print(f"Precision: {precision:.3%}")
print(f"Recall: {recall:.3%}")
print(f"F1 Score: {f1:.3%}")
