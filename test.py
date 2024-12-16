import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

file_path = "C:/Users/raoufbtf/Downloads/imo.csv"
df = pd.read_csv(file_path)

cat_cols = ["a01"]  
cont_cols = [f"a{i:02}" for i in range(3, 35)] 
target_col = 'class'

label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])

class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping des classes :", class_mapping)

scaler = StandardScaler()
df[cont_cols] = scaler.fit_transform(df[cont_cols])

X = df[cat_cols + cont_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)



model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  
    ])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
sample = np.array([
    [1] + [0.5] * 32, 
    [0] + [-0.3] * 32, 
    [1] + [1.2] * 32    
    ])

sample[:, 1:] = scaler.transform(sample[:, 1:])  
predictions = model.predict(sample)
predicted_classes = predictions.argmax(axis=1)  
predicted_labels = label_encoder.inverse_transform(predicted_classes)
for i, predicted_label in enumerate(predicted_labels):
    print(f"Échantillon {i + 1} : Classe prédite -> {predicted_label}")
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title("Accuracy over epochs")
plt.show()