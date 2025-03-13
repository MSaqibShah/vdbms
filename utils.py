# vector_db/utils.py
import numpy as np
import json

def normalize_vector(vector):
    """Normalizes a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm

def load_json(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_json(data, filepath):
    """Saves JSON data to a file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)