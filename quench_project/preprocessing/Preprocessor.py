from .DataLoader import DataLoader
from .feature_scaling import feature_scaling
from .Visualiziation import visualize_sequence, plot_temperature_histograms, plot_label_distribution_pie
import numpy as np

class Preprocessor:

    def __init__(self, normalize=False, reshape_to_2d=True):
        """
        Args:
            normalize (bool): se normalizzare i dati
            reshape_to_2d (bool): se rimodellare ogni frame in (15, 15)
        """
        self.normalize = normalize
        self.reshape_to_2d = reshape_to_2d
        self.min = None
        self.max = None
        self.data = None
        self.path = None
        self.normalization_method = None

    def load_data(self, path):

        dataloader = DataLoader(path)
        self.data = dataloader.get_data()
        self.path = path

        if self.reshape_to_2d:
            self.data['sequence'] = self.data['sequence'].apply(np.array)
            self.data['sequence'] = self.data['sequence'].apply(lambda x: np.reshape(x, (24, 15, 15)))
            self.data['sequence'] = self.data['sequence'].apply(lambda x: np.transpose(x, (0, 2, 1)))
            

        return self.data
    
    def visualize_sequence(self,data, quenched=False, debug=False):
        """
        Wrapper per la funzione `visualize_sequence` per utilizzarla come metodo
        della classe `Preprocessor`.

        quenched: bool
            Se True, mostra una sequenza con almeno un quench (label 1), altrimenti mostra una sequenza senza quench (label 0).
        debug: bool
            Se True, stampa informazioni di debug sul quench.
  
        """
        visualize_sequence(data, quenched=quenched, debug=debug)
        
    def more_data_exploration(self):
        
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        
        plot_temperature_histograms(self.data)
        plot_label_distribution_pie(self.data)
    
    def apply_normalization(self,normalization_method='minmax'):

        self.normalization_method = normalization_method
        self.normalize = True

        if not self.reshape_to_2d:
            raise ValueError("Data must be reshaped to 2D before normalization.")
        
        if normalization_method not in ['minmax', 'standard']:
            raise ValueError("Normalization method must be 'minmax' or 'standard'.")

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        feature_scaling(self.data,self.normalization_method)
        
        
        return None
    





