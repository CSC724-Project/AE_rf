# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==========================
# 1. Load the Dataset
# ==========================
print("Step 1: Loading the dataset")
# Replace the filename with the correct path if necessary.
df = pd.read_csv('beegfs_test_results4.csv')
print("Original DataFrame shape:", df.shape)
print("Columns in the dataset:", df.columns.tolist())
print("-" * 50)

# ==========================
# 2. Remove Unnecessary Columns
# ==========================
print("Step 2: Removing the 'error_message' column (if present)")
if 'error_message' in df.columns:
    df = df.drop('error_message', axis=1)
    print("Column 'error_message' removed.")
else:
    print("Column 'error_message' not found in the dataset.")
print("DataFrame shape after removal:", df.shape)
print("Remaining columns:", df.columns.tolist())
print("-" * 50)

# ==========================
# 3. Select Numeric Columns
# ==========================
print("Step 3: Selecting numeric columns for PCA")
# Select only the numeric columns from the DataFrame.
numeric_df = df.select_dtypes(include=[np.number])
print("Numeric columns:", numeric_df.columns.tolist())
print("Shape of numeric data:", numeric_df.shape)
print("-" * 50)

# ==========================
# 4. Handling Missing Values
# ==========================
print("Step 4: Checking and handling missing values")
print("Missing values in each numeric column before imputation:")
print(numeric_df.isnull().sum())
# Impute missing values with the mean of each column.
numeric_df = numeric_df.fillna(numeric_df.mean())
print("Missing values after imputation:")
print(numeric_df.isnull().sum())
print("-" * 50)

# ==========================
# 5. Standardize the Data
# ==========================
print("Step 5: Standardizing the data")
scaler = StandardScaler()
# Fit and transform the data to have zero mean and unit variance.
data_scaled = scaler.fit_transform(numeric_df)
print("Data scaling complete. Sample of scaled data:")
print(data_scaled[:5])  # Print first 5 rows of the scaled data
print("-" * 50)

# ==========================
# 6. Perform PCA
# ==========================
print("Step 6: Performing PCA")
# Initialize PCA to reduce to 2 components.
pca = PCA(n_components=2)
# Fit PCA on the standardized data.
principal_components = pca.fit_transform(data_scaled)
# Create a DataFrame with the principal component scores.
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print("PCA complete.")
print("Explained variance ratio for PC1 and PC2:", pca.explained_variance_ratio_)
print("First 5 rows of principal component scores:")
print(principal_df.head())
print("-" * 50)

# ==========================
# 7. Examine PCA Loadings
# ==========================
print("Step 7: Examining PCA loadings (feature contributions)")
# The loadings (components) tell us the weight of each original feature in the principal components.
loadings = pd.DataFrame(pca.components_.T, index=numeric_df.columns, columns=['PC1', 'PC2'])
print("PCA loadings:")
print(loadings)
print("-" * 50)

# ==========================
# 8. Visualize the PCA Results
# ==========================
print("Step 8: Visualizing PCA results")
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot of beegfs_test_results4.csv')
plt.grid(True)
plt.show()