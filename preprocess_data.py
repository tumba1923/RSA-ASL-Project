import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = "asl_data"
GESTURES = ["Hello", "Thank You", "Yes", "No", "I Love You", "Help", "Please", "Sorry", "Stop"]

sequences, labels = [], []

for label, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(DATA_PATH, gesture)
    for file in os.listdir(gesture_path):
        sequence = np.load(os.path.join(gesture_path, file))
        sequences.append(sequence)
        labels.append(label)

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(GESTURES))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded: {X.shape[0]} samples, {len(GESTURES)} classes")
