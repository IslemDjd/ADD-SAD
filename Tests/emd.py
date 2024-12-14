import numpy as np
import pandas as pd

# Step 1: Define the matrix A
A = np.array([
    [1, 3],
    [2, 5],
    [3, 1]
])

# Step 2: Center the data
mean = A.mean(axis=0)
A_centered = A - mean
print("Centered Matrix:\n", A_centered)

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(A_centered.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 4: Calculate the correlation matrix
std_dev = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
print("\nCorrelation Matrix:\n", corr_matrix)

# Step 5: Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 6: Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nSorted Eigenvalues:\n", eigenvalues)
print("\nSorted Eigenvectors:\n", eigenvectors)

# Step 7: Calculate the explained variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nExplained Variance Ratio:\n", explained_variance_ratio)

# Step 8: Conclusions
print("\nConclusions:")
for i, evr in enumerate(explained_variance_ratio):
    print(f"Principal Component {i+1}: {evr * 100:.2f}% of the variance explained")