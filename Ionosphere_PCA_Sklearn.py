import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('./DataSet/ionosphere.arff.csv')  # Replace with your actual file path

# Extract only numerical data
numerical_data = data.select_dtypes(include=[np.number])

# Handle missing or infinite values
if numerical_data.isnull().values.any():
    print("Missing values detected. Replacing with column means.")
    numerical_data = numerical_data.fillna(numerical_data.mean())

if np.isinf(numerical_data.values).any():
    print("Infinite values detected. Replacing with column means.")
    numerical_data = numerical_data.replace([np.inf, -np.inf], np.nan)
    numerical_data = numerical_data.fillna(numerical_data.mean())

# Verify no NaN or Inf values exist
assert numerical_data.isnull().sum().sum() == 0, "NaN values remain in the data."
assert np.isinf(numerical_data.values).sum() == 0, "Infinite values remain in the data."

# Check for zero variance columns
zero_variance_cols = numerical_data.columns[numerical_data.std() == 0]
if len(zero_variance_cols) > 0:
    print(f"Zero variance columns detected: {zero_variance_cols}. Dropping them.")
    numerical_data = numerical_data.drop(columns=zero_variance_cols)

# Step 2: Standardize the data (Z-score normalization)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(numerical_data)

# Step 3: Apply PCA
k = 8  # Number of principal components to retain
pca = PCA(n_components=k)
projected_data = pca.fit_transform(standardized_data)

# Display results
print("\nProjected Data (first 5 rows):")
print(projected_data[:5])

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print("\nExplained Variance Ratio (per component):")
print(explained_variance_ratio)
print("\nCumulative Explained Variance Ratio:")
print(cumulative_variance_ratio)
