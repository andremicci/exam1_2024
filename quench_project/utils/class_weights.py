
import numpy as np
import torch

def get_class_weights(dataset):
    """
    Calculate class weights for CrossEntropyLoss from data.num_quench (assumed integer labels starting from 1).
    Missing classes are assigned weight 0. Class indices are converted to 0-based.
    Args:
        dataset (torch.utils.data.Dataset): Dataset containing the data with 'num_quench' labels.
    Returns:
        torch.Tensor: Tensor of class weights, where index corresponds to class label (0-based).
                      Missing classes have weight 0.

    """
    # Etichette presenti e relative frequenze
    arr=dataset.dataset.__getitem__([np.arange(0, len(dataset))])[1].numpy().reshape(-1)
    values, counts = np.unique(arr, return_counts=True)

    # Calcolo pesi: più rara è la classe, più alto il peso
    weights = counts.max() / counts
    num_classes = int(np.max(values))   

    # Inizializza tutti i pesi a 0
    class_weights = torch.zeros(num_classes, dtype=torch.float32)

    # Assegna i pesi calcolati (shiftati: class 1 → index 1, etc.)
    for value, weight in zip(values-1, weights):
        class_idx = int(value)  
        class_weights[class_idx] = weight  

    return class_weights