import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


# Step 1: Load the data
file_path = 'iris.arff.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Inspect the data
# print("Dataset Head:")
# print(data.head())

# Step 2: Remove non-numeric columns (e.g., 'class')
data = data.select_dtypes(include=[np.number])

# Step 3: Standardize the data (mean=0, std=1)
mean = data.mean(axis=0)
std = data.std(axis=0)
data_standardized = (data - mean) / std


print("\n Standarized Data :",data_standardized)
print(data_standardized)
print("\n Standarized Data Mean :")
print(data_standardized.mean(axis=0))
print("\n Standarized Data Std :")
print(data_standardized.std(axis=0))

# Step 4: Compute the Covariance Matrix
cov_matrix = np.cov(data_standardized.T)  # Transpose is needed for proper covariance calculation
print("\n Covariance Matrix :")
print(cov_matrix)

# Step 5: Calculate Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Step 7: Calculate Explained Variance
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)
print("\nExplained Variance Ratio:")
print(explained_variance)
print("\nCumulative Explained Variance:")
print(cumulative_variance)

# Step 8: Project the data onto the top k principal components
k = 2  # Number of principal components to keep
top_k_eigenvectors = eigenvectors[:, :k]
projected_data = np.dot(data_standardized, top_k_eigenvectors)
print("\n Projected Data : ")
print(projected_data)

# Step 9: Visualize the Results
# plt.figure(figsize=(8, 6))
# plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.7)
# plt.title('Projection onto First Two Principal Components')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid()
# plt.show()

# Optional: Save the projected data
# projected_df = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(k)])
# output_file = '/mnt/data/manual_pca_results.csv'
# projected_df.to_csv(output_file, index=False)
# print(f"PCA results saved to {output_file}")




