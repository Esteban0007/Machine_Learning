# Imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Constant
DATA_PATH = os.path.join("MP_Data")
ACTIONS = np.array(["hello", "thanks", "iloveyou"])

# Variables
sequence_length = 30
label_map = {label: num for num, label in enumerate(ACTIONS)}
label_map
{"Hola": 0, "gracias": 1, "te quiero": 2, "yo": 4, "Esteban": 5, "tu": 6}
sequences, labels = [], []

# Load data and preprocess it
for action in ACTIONS:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(
                    DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                )
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
np.array(sequences).shape
(180, 30, 1662)
np.array(labels).shape
(180,)
X = np.array(sequences)
X.shape
(180, 30, 1662)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
y_test.shape

# Save data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
