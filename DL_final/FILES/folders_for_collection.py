# IMPORTS ----------------------------------------------------------------
import os
import numpy as np

# CONSTANTS ----------------------------------------------------------------
DATA_PATH = os.path.join("MP_Data")

# VARIABLES ----------------------------------------------------------------
actions = np.array(["hello", "thanks", "iloveyou"])


no_sequences = 30
sequence_length = 30
start_folder = 30

# MAIN ----------------------------------------------------------------

# Create directories
for action in actions:
    dir_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    files = os.listdir(dir_path)
    if files:
        dirmax = np.max(np.array(files).astype(int))
    else:
        dirmax = 0  # Empezar desde 0 si no hay archivos

    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax + sequence)))
        except Exception as e:
            print(f"Error al crear el directorio: {e}")
