
from .Visualiziation import visualize_sequence, plot_temperature_histograms, plot_label_distribution_pie
import numpy as np

class DataExplorator:

    def __init__(self, data):
        self.data = data
  
    
    def visualize_sequence(self,quenched=False, debug=False):  
        visualize_sequence(self.data, quenched=quenched, debug=debug)
        
    def more_data_exploration(self):
        
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
          
        plot_temperature_histograms(self.data)
        plot_label_distribution_pie(self.data)
    

