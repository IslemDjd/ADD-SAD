#Ce fichier marche pour les deux Iris et Ionosphere
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# data = pd.read_csv('./DataSet/iris.arff.csv')
data = pd.read_csv('./DataSet/ionosphere.arff.csv')

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
print(standardized_data)

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

# Step 6: Compute the explained variance ratio
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

# Step 7: Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Step 8: Project the data onto the top C principal components
C = 8
C_eigenvectors = sorted_eigenvectors[:, :C]
projected_data = np.dot(standardized_data, C_eigenvectors)

print(projected_data)
# Step 9: Prepare the plot
if 'class' in data.columns:
    labels = data['class'].to_numpy()  # Convert to NumPy array for indexing compatibility
    unique_labels = np.unique(labels)  # Find unique class labels

    # Assign unique colors to each label
    colors = plt.colormaps['tab10']  # Fetch colormap
    color_map = {label: colors(i / len(unique_labels)) for i, label in enumerate(unique_labels)}
else:
    raise ValueError("No 'class' column found; cannot color-code the plot.")

# Step 10: Project data onto the first two principal components for visualization
projected_data_2D = projected_data[:, :2]  # Use only the first two principal components

plt.figure(figsize=(8, 6))

# Scatter plot for the first two principal components
for label in unique_labels:
    label_points = projected_data_2D[labels == label]  # Filter points by label
    plt.scatter(
        label_points[:, 0],  # PC1
        label_points[:, 1],  # PC2
        color=color_map[label],  # Get the color for this label
        alpha=0.7,  # Transparency
        edgecolor='k',  # Black edge for points
        label=label  # Class label for the legend
    )

# Add titles and axis labels
plt.title("Projection onto First Two Principal Components (Colored by Class)", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.grid(True)
plt.legend(loc='best', title="Classes")

# Save the plot
plt.savefig("Ionosphere_ACP_Result/Ionosphere_ACP.png", dpi=300)
print("Plot saved as 'Ionosphere_ACP_Result/Ionosphere_ACP.png'.")

# Show the plot
plt.show()
