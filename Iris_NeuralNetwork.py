import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the dataset
file_path = './DataSet/iris.arff.csv'  # Update with the correct path
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Assume 'class' is the label column
if 'class' in data.columns:
    X = data.drop(columns=['class'])
    y = data['class']
else:
    raise ValueError("Error: 'class' column not found in the dataset.")

# Convert categorical labels to numerical if necessary
if y.dtype == 'object':
    encoder = OneHotEncoder(sparse_output=False)  # Updated for compatibility with scikit-learn >= 1.2
    y = encoder.fit_transform(y.values.reshape(-1, 1))

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the neural network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(y_train.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Step 6: Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
y_true_classes = np.argmax(y_test, axis=1)  # True class indices from one-hot encoded labels

# Step 7: Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Step 8: Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
