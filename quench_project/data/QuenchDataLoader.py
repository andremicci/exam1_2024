import json
import os
import pandas as pd
import numpy as np


class QuenchDataLoader:

    def __init__(self, path, reshape_to_2d=True):
        """
        Args:
            path (str): percorso del file JSON
        """
        self.path = path
        self.data = None
        self.reshape_to_2d = True

    def load_data(self):
        """
        Carica i dati dal file JSON.
        """
        if os.path.exists(self.path):
            with open(self.path, "r") as file:
                self.data = json.load(file)
            return self.data
        else:
            raise FileNotFoundError(f"File not found: {self.path}")
        

    def get_data(self):
        """
        Restituisce i dati caricati.
        """
        self.load_data()
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        self.data = pd.DataFrame(self.data)

        if self.reshape_to_2d:
            self.data['sequence'] = self.data['sequence'].apply(np.array)
            self.data['sequence'] = self.data['sequence'].apply(lambda x: np.reshape(x, (24, 15, 15)))
            self.data['sequence'] = self.data['sequence'].apply(lambda x: np.transpose(x, (0, 2, 1)))
        
        
        self.data['num_quench']= self.data[self.data.label==1]['quench'].apply(lambda x: len(x))
        return self.data
    
    
    
        
        