import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
file_path = './DataSet/iris.arff.csv'  # Adjust the path if necessary
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
    exit()

# Step 2: Separate non-numeric columns (e.g., 'class')
if 'class' in data.columns:
    labels = data['class']  # Use 'class' column for coloring
    data = data.drop(columns=['class'])  # Remove the 'class' column for PCA
else:
    labels = None

# Step 3: Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Step 4: Perform PCA using the sklearn library
k = 2  # Number of principal components to retain
pca = PCA(n_components=k)
projected_data = pca.fit_transform(data_standardized)

# Step 5: Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Explained Variance:", cumulative_variance)

# Step 6: Visualize the results with color coding
plt.figure(figsize=(8, 6))

if labels is not None:
    # Map labels to unique colors
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
    # Use a continuous color map if no categorical labels are available
    plt.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=projected_data[:, 0],  # Use first principal component for coloring
        cmap='viridis',
        alpha=0.7
    )

plt.title('Projection onto First Two Principal Components (sklearn PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()

# Save the plot
plot_file = 'Iris_PCA_Sklearn/iris_sklearn.png'
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")

# Show the plot
plt.show()

# Optional: Save the projected data
projected_df = pd.DataFrame(projected_data, columns=[f'PC{i+1}' for i in range(k)])
if labels is not None:
    projected_df['label'] = labels
output_file = 'Iris_PCA_Sklearn/IrisACP_sklearn.csv'
projected_df.to_csv(output_file, index=False)
print(f"PCA results saved to {output_file}")
