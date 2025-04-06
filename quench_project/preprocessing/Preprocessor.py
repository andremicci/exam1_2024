from .DataLoader import DataLoader
from .Visualiziation import visualize_sequence
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

    def load_data(self, path):

        dataloader = DataLoader(path)
        self.data = dataloader.get_data()
        self.path = path

        if self.reshape_to_2d:
            self.data['sequence'] = self.data['sequence'].apply(np.array)
            self.data['sequence'] = self.data['sequence'].apply(lambda x: np.reshape(x, (24, 15, 15)))

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
    
    def normalize_data(self):
        """
        Normalizza i dati.
        """
        '''
        if self.normalize:
            self.min = np.min(data)
            self.max = np.max(data)
            data = (data - self.min) / (self.max - self.min + 1e-8)

        '''
        
        return None
    





