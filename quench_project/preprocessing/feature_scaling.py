import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale


def feature_scaling(data,normalization_method):
      
    normalization_method = normalization_method
    sequences = np.stack(data['sequence'].values).reshape(-1, 15*15) # shape (N*24, 15*15)

    if normalization_method == 'minmax':
        
        normalized_data = minmax_scale(sequences).reshape(-1, 24, 15,15) # normalize and return to shape (N, 24, 15*15)
        data['sequence'] = [row for row in normalized_data]

        

    
    if normalization_method == 'standard':
        
        normalized_data = scale(sequences).reshape(-1, 24, 15,15) 
        data['sequence'] = [row for row in normalized_data]