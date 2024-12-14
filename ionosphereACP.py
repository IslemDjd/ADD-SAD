import pandas as pd
import numpy as np

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
standardized_data = (numerical_data - numerical_data.mean()) / numerical_data.std()

# Verify no NaN or Inf values exist after standardization
assert np.isfinite(standardized_data.values).all(), "Non-finite values in standardized data."

# Step 3: Compute the covariance matrix
covariance_matrix = np.cov(standardized_data.T)

# Step 4: Perform Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 5: Sort Eigenvalues and Eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# print("\nEigenvalues:")
# print(sorted_eigenvalues)
# print("\nEigenvectors:")
# print(sorted_eigenvectors)

# # Step 6: Project the data onto the top k components
# k = 8  # Number of principal components to retain
# top_eigenvectors = sorted_eigenvectors[:, :k]
# projected_data = standardized_data.dot(top_eigenvectors)
#
# # Convert projected data to a DataFrame for easier handling
# projected_df = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(k)])


# Step 8: Project the data onto the top k principal components
k = 8
# Number of principal components to keep
top_k_eigenvectors = eigenvectors[:, :k]
projected_data = np.dot(standardized_data, top_k_eigenvectors)

# Display results
print("\n Projected Data (first 5 rows):")
print(projected_data[:5])

