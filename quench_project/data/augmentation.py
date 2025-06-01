import numpy as np

def rotate_90(seq):
    # ruota di 90° ogni frame nella sequenza
    return np.array([np.rot90(frame, k=1) for frame in seq])

def rotate_180(seq):
    # ruota di 180° ogni frame nella sequenza
    return np.array([np.rot90(frame, k=2) for frame in seq])

def rotate_270(seq):
    # ruota di 270° ogni frame nella sequenza
    return np.array([np.rot90(frame, k=3) for frame in seq])


def flip_horizontal(seq):
    # flip orizzontale per ogni frame
    return np.array([np.fliplr(frame) for frame in seq])

def flip_vertical(seq):
    # flip verticale per ogni frame
    return np.array([np.flipud(frame) for frame in seq])

def transpose(seq):
    # trasponi ogni frame nella sequenza
    return np.array([np.transpose(frame) for frame in seq])

def flip_diagonal_secondary(seq):
    return np.array([np.flipud(np.fliplr(np.transpose(frame))) for frame in seq])



def time_reversal(seq):
    # inverti l'ordine temporale
    return seq[::-1]

def reverse_and_flip(seq):
    # applica flip orizzontale e inversione temporale
    return flip_horizontal(time_reversal(seq))

def shuffle(seq, fraction=0.3):
    seq = seq.copy()
    n = int(len(seq) * fraction)
    idx = np.random.choice(len(seq), size=n, replace=False)
    np.random.shuffle(seq[idx])
    return seq


def add_gaussian_noise(seq, mean=0, std=0.01):
    noise = np.random.normal(mean, std, seq.shape)
    return seq + noise
