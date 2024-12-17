import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = './DataSet/iris.arff.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
    exit()

# Step 2: Separate non-numeric columns (e.g., 'class')
if 'class' in data.columns:
    labels = data['class']
    data = data.drop(columns=['class'])  # Remove the 'class' column for PCA
else:
    labels = None

# Step 3: Standardize the data (mean=0, std=1)
mean = data.mean(axis=0)
std = data.std(axis=0)
data_standardized = (data - mean) / std

# Step 4: Compute the Covariance Matrix
cov_matrix = np.cov(data_standardized.T) 
print(cov_matrix)
# Step 5: Calculate Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 6: Sort Eigenvalues and Eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 7: Calculate Explained Variance
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

# Step 8: Project the data onto the top k principal components
k = 2 
top_k_eigenvectors = eigenvectors[:, :k]
projected_data = np.dot(data_standardized, top_k_eigenvectors)

# Step 9: Visualize the Results with Color Coding
plt.figure(figsize=(8, 6))

if labels is not None:
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        plt.scatter(
            projected_data[labels == label, 0],
            projected_data[labels == label, 1],
            label=label,
            alpha=0.7,
            color=color
        )
else:
    plt.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=projected_data[:, 0],
        cmap='viridis',
        alpha=0.7
    )

plt.title('Projection onto First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()

# Save the plot
plot_file = 'Iris_ACP_Result/iris.png'
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")

# Show the plot
plt.show()


# Optional: Save the projected data
projected_df = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(k)])
if labels is not None:
    projected_df['label'] = labels
output_file = 'Iris_ACP_Result/IrisACP.csv'
projected_df.to_csv(output_file, index=False)
print(f"PCA results saved to {output_file}")
