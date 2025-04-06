import json
import os
import pandas as pd


class DataLoader:

    def __init__(self, path):
        """
        Args:
            path (str): percorso del file JSON
        """
        self.path = path
        self.data = None

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
        return self.data
    
    
    
        
        