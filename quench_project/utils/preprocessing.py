import numpy as np

def load_and_preprocess(filepath):
    # Placeholder per caricamento dati (da completare con caricamento reale)
    data = np.load(filepath, allow_pickle=True)
    sequences = []
    labels = []
    for sample in data:
        seq = sample["sequence"].reshape(24, 15, 15)
        sequences.append(seq)
        labels.append(sample["label"])
    return np.array(sequences), np.array(labels)
